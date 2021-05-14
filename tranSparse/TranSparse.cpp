#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <string>
#include <algorithm>
#include <pthread.h>

using namespace std;

const float pi = 3.141592653589793238462643383;
int bern = 0;
int tranSparseThreads = 8;
int tranSparseTrainTimes = 1000;
int nbatches = 100;
int dimension = 50;
int dimensionR = 50;
float tranSparseAlpha = 0.001;
float margin = 1;

string inPath = "./data/";
string outPath = "./out/";

int *lefHead, *rigHead;
int *lefTail, *rigTail;

struct Triple {
	int h, r, t;
};

struct cmp_head {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}
};  //比较头实体，头实体若相等就比关系，关系相等就比尾实体

struct cmp_tail {
	bool operator()(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}
};  //比较尾实体，尾实体若相等就比关系，关系相等就比头实体

struct cmp_list {
	int minimal(int a,int b) {
		if (a < b) return b;
		return a;
	}
	bool operator()(const Triple &a, const Triple &b) {
		return (minimal(a.h, a.t) < minimal(b.h, b.t));
	}
};   //将三元组a,b里面，头尾实体里较大的拿出来去比较

Triple *trainHead, *trainTail, *trainList;

/*
	There are some math functions for the program initialization.
*/

unsigned long long *next_random;  //转换数组next_random中index为id的值

unsigned long long randd(int id) {  //转换next_random索引为id的值并返回
	next_random[id] = next_random[id] * (unsigned long long)25214903917 + 11;
	return next_random[id];  
}

int rand_max(int id, int x) {  //处理上面数组next_random中下面为id小于x的数，小于则返回x。
	int res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

float rand(float min, float max) { //产生一个min和max之间的伪随机数
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) { //高斯分布，返回x的概率密度函数
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float randn(float miu,float sigma, float min ,float max) {
	float x, y, dScope;
	do {
		x = rand(min,max);  //min与max之间的伪随机数
		y = normal(x,miu,sigma);  //x的概率密度，即min与max之间伪随机数的概率密度
		dScope=rand(0.0,normal(miu,miu,sigma));  //0与均值miu的概率密度，之间的伪随机数
	} while (dScope > y);  
	return x;
}

void norm(float *con, int dimension) {  //向量标准化
	float x = 0;
	for (int  ii = 0; ii < dimension; ii++)
		x += (*(con + ii)) * (*(con + ii));
	x = sqrt(x);
	if (x>1)
		for (int ii=0; ii < dimension; ii++)
			*(con + ii) /= x;
}

void norm(float *con, float *matrix, int *sparse) {  //稀疏矩阵标准化
	float tmp, x = 0;
	int last = 0, lastM = 0;
	for (int ii = 0; ii < dimensionR; ii++) {
		tmp = 0;
		for (int i = sparse[last]; i >= 1; i--)
			tmp += matrix[lastM + sparse[last + i]] * con[sparse[last + i]];
		x += tmp * tmp;
		last += sparse[last] + 1;
		lastM += dimension;
	}
	if (x > 1) {
		float lambda = 1;
		last = 0; lastM = 0;
		for (int ii = 0; ii < dimensionR; ii++) {
			tmp = 0;
			for (int jj = sparse[last]; jj >= 1; jj--)
				tmp += matrix[lastM + sparse[last + jj]] * con[sparse[last + jj]];
			tmp = tmp + tmp;
			for (int jj = sparse[last]; jj >= 1; jj--) {
				matrix[lastM + sparse[last + jj]] -= tranSparseAlpha * lambda * tmp * con[sparse[last + jj]];
				con[sparse[last + jj]] -= tranSparseAlpha * lambda * tmp * matrix[lastM + sparse[last + jj]];
			}
			last += sparse[last] + 1;
			lastM += dimension;
		}
	}
}

int relationTotal, entityTotal, tripleTotal;
int *freqRel, *freqEnt;
float *left_mean, *right_mean;
float *relationVec, *entityVec, *matrixHead, *matrixTail;
float *relationVecDao, *entityVecDao, *matrixHeadDao, *matrixTailDao;
float *tmpValue;
int *sparse_id_l, *sparse_id_r, *sparse_pos_l, *sparse_pos_r;

void norm(int h, int t, int r, int j, int tip) { //j表示负样本
		norm(relationVecDao + dimensionR * r, dimensionR);
		norm(entityVecDao + dimension * h, dimension);
		norm(entityVecDao + dimension * t, dimension);
		norm(entityVecDao + dimension * j, dimension);
		norm(entityVecDao + dimension * h, matrixHeadDao + dimension * dimensionR * r, sparse_id_l + sparse_pos_l[r]);
		norm(entityVecDao + dimension * t, matrixTailDao + dimension * dimensionR * r, sparse_id_r + sparse_pos_r[r]);
		if (tip == 1)
			norm(entityVecDao + dimension * j, matrixHeadDao + dimension * dimensionR * r, sparse_id_l + sparse_pos_l[r]);
		else
			norm(entityVecDao + dimension * j, matrixTailDao + dimension * dimensionR * r, sparse_id_r + sparse_pos_r[r]);
}

/*
	Read triples from the training file.
*/

void init() {

	FILE *fin;
	int tmp;

	fin = fopen((inPath + "relation2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &relationTotal);
	fclose(fin);

	fin = fopen((inPath + "entity2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &entityTotal);
	fclose(fin);

	relationVec = (float *)calloc(relationTotal * dimensionR * 2 + entityTotal * dimension * 2 + relationTotal * dimension * dimensionR * 4, sizeof(float));
	relationVecDao = relationVec + relationTotal * dimensionR;
	entityVec = relationVecDao + relationTotal * dimensionR;
	entityVecDao = entityVec + entityTotal * dimension;
	matrixHead = entityVecDao + entityTotal * dimension;
	matrixHeadDao = matrixHead + dimension * dimensionR * relationTotal;
	matrixTail = matrixHeadDao + dimension * dimensionR * relationTotal;
	matrixTailDao = matrixTail + dimension * dimensionR * relationTotal;  //定义各向量维数

	freqRel = (int *)calloc(relationTotal + entityTotal, sizeof(int));
	freqEnt = freqRel + relationTotal;  //针对头尾实体稀疏度的分子N，针对关系、实体总数分配内存给freqRel，freqEnt，分别表示关系、实体的频率

	for (int i = 0; i < relationTotal; i++) {
		for (int ii=0; ii < dimensionR; ii++)
			relationVec[i * dimensionR + ii] = randn(0, 1.0 / dimensionR, -6 / sqrt(dimensionR), 6 / sqrt(dimensionR));
	}
	for (int i = 0; i < entityTotal; i++) {
		for (int ii=0; ii < dimension; ii++)
			entityVec[i * dimension + ii] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
		norm(entityVec + i * dimension, dimension);   //关系、实体向量初始化
	}

	for (int i = 0; i < relationTotal; i++)
		for (int j = 0; j < dimensionR; j++)
			for (int k = 0; k < dimension; k++) {
				matrixHead[i * dimension * dimensionR + j * dimension + k] =  randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
				matrixTail[i * dimension * dimensionR + j * dimension + k] =  randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
			}   //稀疏矩阵初始化

	fin = fopen((inPath + "triple2id.txt").c_str(), "r");
	tmp = fscanf(fin, "%d", &tripleTotal);
	trainHead = (Triple *)calloc(tripleTotal * 3, sizeof(Triple));
	trainTail = trainHead + tripleTotal;  //根据训练集大小，分别分配相应的内存空间
	trainList = trainTail + tripleTotal;   //trainList存储三元组，复制给trainhead和traintail
	for (int i = 0; i < tripleTotal; i++) {
		tmp = fscanf(fin, "%d", &trainList[i].h);
		tmp = fscanf(fin, "%d", &trainList[i].t);
		tmp = fscanf(fin, "%d", &trainList[i].r);  //将train2id.txt中的三列数据，分别保存到trainList中
		freqEnt[trainList[i].t]++;  //以trainList[0]的尾实体作为数组freqEnt的下标，对应的值+1
		freqEnt[trainList[i].h]++;  //以trainList[0]的头实体作为数组freqEnt的下标，对应的值+1
		freqRel[trainList[i].r]++;  //以trainList[0]的关系作为数组freqEnt的下标，对应的值+1
		trainHead[i] = trainList[i];
		trainTail[i] = trainList[i];
	}
	fclose(fin);

	sort(trainHead, trainHead + tripleTotal, cmp_head());  //按照head进行排序
	sort(trainTail, trainTail + tripleTotal, cmp_tail());  //按照tail进行排序
	sort(trainList, trainList + tripleTotal, cmp_list());  //按照head和tail中较大的进行排序

	lefHead = (int *)calloc(entityTotal * 4, sizeof(int));  //以实体总数，分配内存空间给lefHead、lefTail、rigTail、righead
	rigHead = lefHead + entityTotal;
	lefTail = rigHead + entityTotal;
	rigTail = lefTail + entityTotal;
	memset(rigHead, -1, sizeof(int)*entityTotal);
	memset(rigTail, -1, sizeof(int)*entityTotal);  //对数组rigHead、rigTail初始化为-1
	//从i=1～trainTotal，ritTail保存的是尾实体ID较小的对应的trainT下标，lefTail保存的是尾实体ID较大的对应的trainT下标
	for (int i = 1; i < tripleTotal; i++) {
		//如果trainTail，第i中的尾实体与i-1中的尾实体不一样
	    //即，如果相邻两个训练尾实体不相同，则以前者尾实体为rigTail的下标，将i-1替换对应的-1
	    // 将后者尾实体为lefTail的下标，将i替换对应的-1
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;  //将i-1赋值给以trainTail[i-1]的尾实体为下标，对应的rigTail值-1
			lefTail[trainTail[i].t] = i;          //将i赋值给以trainTail[i]的尾实体为下标，对应的lefTail值-1
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
	}
	rigHead[trainHead[tripleTotal - 1].h] = tripleTotal - 1;
	rigTail[trainTail[tripleTotal - 1].t] = tripleTotal - 1;

	left_mean = (float *)calloc(relationTotal * 2,sizeof(float));  //为left_mean、right_mean分配实数型的内存，元素个数为relationTotal，大小为REAL
	right_mean = left_mean + relationTotal;
	for (int i = 0; i < entityTotal; i++) {
		for (int j = lefHead[i] + 1; j < rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;   //相邻训练头实体对应的关系不等情况下，对头实体的出边+1
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;  //如果左实体的大小小于等于右实体的大小，则以左实体对应的出边+1
		for (int j = lefTail[i] + 1; j < rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (int i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];   //实体的个数除以对应实体的出边
		right_mean[i] = freqRel[i] / right_mean[i];  //实体的个数除以对应实体的入边，算作该关系link的实体对个数
	}
	for (int i = 0; i < relationTotal; i++)
		for (int j = 0; j < dimensionR; j++)
			for (int k = 0; k < dimension; k++)  //仅将对角线元素赋值为1
				if (j == k) {
					matrixHead[i * dimension * dimensionR + j * dimension + k] = 1;
					matrixTail[i * dimension * dimensionR + j * dimension + k] = 1;
				}
				else {
					matrixHead[i * dimension * dimensionR + j * dimension + k] = 0;
					matrixTail[i * dimension * dimensionR + j * dimension + k] = 0;
				}。 //稀疏矩阵赋值
	
	FILE* f1 = fopen((inPath + "tranSparsedata/entity2vec.bern").c_str(),"r");
	for (int i = 0; i < entityTotal; i++) {
		for (int ii = 0; ii < dimension; ii++)
			tmp = fscanf(f1, "%f", &entityVec[i * dimension + ii]);
		norm(entityVec + i * dimension, dimension);
	}
	fclose(f1);
	FILE* f2 = fopen((inPath + "tranSparsedata/relation2vec.bern").c_str(),"r");
	for (int i=0; i < relationTotal; i++) {
		for (int ii=0; ii < dimension; ii++)
			tmp = fscanf(f2, "%f", &relationVec[i * dimensionR + ii]);
	}
	fclose(f2);

	int numLef, numRig;

	FILE* f_d_l = fopen((inPath + "set_num_l.txt").c_str(), "r");  //分别定义left和right的稀疏度
	fscanf(f_d_l, "%d", &numLef);
	sparse_id_l = (int *)calloc(numLef, sizeof(int));
	sparse_pos_l = (int *)calloc(relationTotal, sizeof(int));
	for (int i = 0; i < numLef; i++)
		tmp = fscanf(f_d_l, "%d", &sparse_id_l[i]);
	for (int i = 0, last = 0; i < relationTotal; i++) {
		sparse_pos_l[i] = last;
		for (int j = 0; j < dimensionR; j++)
			last = last + sparse_id_l[last] + 1;
	}
	fclose(f_d_l);

	FILE* f_d_r = fopen((inPath + "set_num_r.txt").c_str(), "r");
	fscanf(f_d_r, "%d", &numRig);
	sparse_id_r = (int *)calloc(numRig, sizeof(int));
	sparse_pos_r = (int *)calloc(relationTotal, sizeof(int));
	for (int i = 0; i < numRig; i++)
		tmp = fscanf(f_d_r, "%d", &sparse_id_r[i]);
	for (int i = 0, last = 0; i < relationTotal; i++) {
		sparse_pos_r[i] = last;
		for (int j = 0; j < dimensionR; j++)
			last = last + sparse_id_r[last] + 1;
	}
	fclose(f_d_r);
}

/*
	Training process of tranSparse.
*/

int tranSparseLen;
int tranSparseBatch;
float res;

double calc_sum(int e1, int e2, int rel, float *tmp1, float *tmp2) { //计算距离
	int lastM = rel * dimensionR * dimension;
	int last1 = e1 * dimension;
	int last2 = e2 * dimension;
	int lastR = rel * dimensionR;
	int lastl = sparse_pos_l[rel], lastr = sparse_pos_r[rel];
	float sum = 0;
	for (int i = 0; i < dimensionR; i++) {
		tmp1[i] = 0;
		for (int jj = sparse_id_l[lastl]; jj >= 1; jj--) {
			int j = sparse_id_l[lastl+jj];
			tmp1[i] += matrixHead[lastM + j] * entityVec[last1 + j];
		}
		tmp2[i] = 0;
		for (int jj = sparse_id_l[lastr]; jj >= 1; jj--) {
			int j = sparse_id_r[lastr+jj];
			tmp2[i] += matrixTail[lastM + j] * entityVec[last2 + j];
		}
		lastM += dimension;
		lastl += sparse_id_l[lastl] + 1;
		lastr += sparse_id_r[lastr] + 1;
		sum += fabs(tmp1[i] + relationVec[lastR + i] - tmp2[i]);  //计算loss值
	}
	return sum;
}

void gradient(int e1_a, int e2_a, int rel_a, int belta, int same, float *tmp1, float *tmp2) {
	int lasta1 = e1_a * dimension;  //更新梯度，正样本试图缩小梯度，负样本试图扩大梯度
	int lasta2 = e2_a * dimension;
	int lastar = rel_a * dimensionR;
	int lastM = rel_a * dimensionR * dimension;
	int lastl = sparse_pos_l[rel_a], lastr = sparse_pos_r[rel_a];
	float x;
	for (int ii=0; ii < dimensionR; ii++) {
		x = tmp2[ii] - tmp1[ii] - relationVec[lastar + ii];
		if (x > 0)
			x = belta * tranSparseAlpha;
		else
			x = -belta * tranSparseAlpha;
		for (int j = sparse_id_l[lastl]; j >= 1; j--) {
			int jj = sparse_id_l[lastl + j];
			matrixHeadDao[lastM + jj] -=  x * (entityVec[lasta1 + jj]);
			entityVecDao[lasta1 + jj] -= x * matrixHead[lastM + jj];
		}
		for (int j = sparse_id_r[lastr]; j >= 1; j--) {
			int jj = sparse_id_r[lastr + j];
			matrixTailDao[lastM + jj] -=  x * (-entityVec[lasta2 + jj]);
			entityVecDao[lasta2 + jj] += x * matrixTail[lastM + jj];
		}
		relationVecDao[lastar + ii] -= same * x;
		lastM = lastM + dimension;
		lastl += sparse_id_l[lastl] + 1;
		lastr += sparse_id_r[lastr] + 1;
	}
}
//计算距离更新梯度
void train_kb(int e1_a, int e2_a, int rel_a, int e1_b, int e2_b, int rel_b, float *tmp) {
	float sum1 = calc_sum(e1_a, e2_a, rel_a, tmp, tmp + dimensionR);
	float sum2 = calc_sum(e1_b, e2_b, rel_b, tmp + dimensionR * 2, tmp + dimensionR * 3);
	if (sum1 + margin > sum2) { //不满足条件则需要更新梯度
		res += margin + sum1 - sum2;
		gradient(e1_a, e2_a, rel_a, -1, 1, tmp, tmp + dimensionR);
    	gradient(e1_b, e2_b, rel_b, 1, 1, tmp + dimensionR * 2, tmp + dimensionR * 3);
	}
}

int corrupt_head(int id, int h, int r) {  //根据相同的h返回一个假的样本t，获取三元组中相同h对应r
	int lef, rig, mid, ll, rr;
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {  //则该值不只一个
		mid = (lef + rig) >> 1;  //除2
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;  //r值对应的index
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));  //生成一个小于entityTotal - (rr - ll + 1)的随机数
	if (tmp < trainHead[ll].t) return tmp;  //小于初始t 直接返回
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

int corrupt_tail(int id, int t, int r) {
	int lef, rig, mid, ll, rr;
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	int tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}
// 接受线程id作为输入，调用corrupt生成正负样本，train_kb进行训练
void* tranSparsetrainMode(void *con) {
	int id, i, j, pr, tip;
	id = (unsigned long long)(con);  //补0即可
	next_random[id] = rand();
	float *tmp = tmpValue + id * dimensionR * 4;
	for (int k = tranSparseBatch / tranSparseThreads; k >= 0; k--) { // 一个batch训练的样本数按照线程均分
		i = rand_max(id, tranSparseLen);	 // 生成一个样本随机的样本id,i为生成的随机数
		if (bern)
			pr = 1000*right_mean[trainList[i].r]/(right_mean[trainList[i].r]+left_mean[trainList[i].r]);
		else
			pr = 500;  //一半的概率1/2决定生成 伪head tail
		if (randd(id) % 1000 < pr) { // 选择正、负样本作为训练输入
			j = corrupt_head(id, trainList[i].h, trainList[i].r);
			tip = 0;
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, trainList[i].h, j, trainList[i].r, tmp);
		} else {
			j = corrupt_tail(id, trainList[i].t, trainList[i].r);
			tip = 1;
			train_kb(trainList[i].h, trainList[i].t, trainList[i].r, j, trainList[i].t, trainList[i].r, tmp);
		}
		norm(trainList[i].h, trainList[i].t, trainList[i].r, j, tip);  //标准化
	}
	pthread_exit(NULL);
}

void* train_tranSparse(void *con) {
	tranSparseLen = tripleTotal;
	tranSparseBatch = tranSparseLen / nbatches;
	next_random = (unsigned long long *)calloc(tranSparseThreads, sizeof(unsigned long long));
	tmpValue = (float *)calloc(tranSparseThreads * dimensionR * 4, sizeof(float));
	memcpy(relationVecDao, relationVec, dimensionR * relationTotal * sizeof(float));
	memcpy(entityVecDao, entityVec, dimension * entityTotal * sizeof(float));
	memcpy(matrixHeadDao, matrixHead, dimension * relationTotal * dimensionR * sizeof(float));
	memcpy(matrixTailDao, matrixTail, dimension * relationTotal * dimensionR * sizeof(float));
	for (int epoch = 0; epoch < tranSparseTrainTimes; epoch++) {
		res = 0;
		for (int batch = 0; batch < nbatches; batch++) {
			pthread_t *pt = (pthread_t *)malloc(tranSparseThreads * sizeof(pthread_t));
			for (long a = 0; a < tranSparseThreads; a++)
				pthread_create(&pt[a], NULL, tranSparsetrainMode,  (void*)a);
			for (long a = 0; a < tranSparseThreads; a++)
				pthread_join(pt[a], NULL);
			free(pt);
			memcpy(relationVec, relationVecDao, dimensionR * relationTotal * sizeof(float));
			memcpy(entityVec, entityVecDao, dimension * entityTotal * sizeof(float));
			memcpy(matrixHead, matrixHeadDao, dimension * relationTotal * dimensionR * sizeof(float));
			memcpy(matrixTail, matrixTailDao, dimension * relationTotal * dimensionR * sizeof(float));
		}
		printf("epoch %d %f\n", epoch, res);
	}
}

/*
	Get the results of tranSparse.
*/

void out_tranSparse() {
		FILE* f2 = fopen((outPath + "relation2vec.vec").c_str(), "w");
		FILE* f3 = fopen((outPath + "entity2vec.vec").c_str(), "w");
		for (int i = 0; i < relationTotal; i++) {
			int last = dimension * i;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f2, "%.6f\t", relationVec[last + ii]);
			fprintf(f2,"\n");
		}
		for (int  i = 0; i < entityTotal; i++) {
			int last = i * dimension;
			for (int ii = 0; ii < dimension; ii++)
				fprintf(f3, "%.6f\t", entityVec[last + ii] );
			fprintf(f3,"\n");
		}
		fclose(f2);
		fclose(f3);
		FILE* f1 = fopen((outPath + "A.vec").c_str(),"w");
		for (int i = 0; i < relationTotal; i++)
			for (int jj = 0; jj < dimension; jj++) {
				for (int ii = 0; ii < dimensionR; ii++)
					fprintf(f1, "%f\t", matrixHead[i * dimensionR * dimension + jj + ii * dimension]);
				fprintf(f1,"\n");
			}
		for (int i = 0; i < relationTotal; i++)
			for (int jj = 0; jj < dimension; jj++) {
				for (int ii = 0; ii < dimensionR; ii++)
					fprintf(f1, "%f\t", matrixTail[i * dimensionR * dimension + jj + ii * dimension]);
				fprintf(f1,"\n");
			}
		fclose(f1);
}

/*
	Main function
*/

int main() {
	init();
	train_tranSparse(NULL);
	out_tranSparse();
	return 0;
}
