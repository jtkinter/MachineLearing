#include "knn_algorithm.h"

#include "datdeal.h"
#include <numeric>
#include <queue>


// *****************************************************************************
// *******************************基本算法**************************************
// *****************************************************************************

// 1.获取最佳的K值
std::pair<int, std::vector<std::vector<int>>> getBestK(std::vector<Data>& data)
{
	if (data.empty())
		throw std::invalid_argument("Invalid input for getBestK: data is null");

	std::vector<double> k;
	std::vector<std::vector<std::vector<int>>> lst;
	for (int i = 3; i < sqrt(data.size()); i += 2)
	{
		auto [res, list] = kFoldCrossVaild(data, 10, i);
		k.push_back(res);
		lst.push_back(list);
	}
	int offset = max_element(k.begin(), k.end()) - k.begin();
	int knn_k = offset * 2 + 3;
	std::cout << "k = " << knn_k << " accuracy = " << k[offset] << std::endl;

	return std::make_pair(knn_k, lst[offset]);
}

#if 0

#include <future>
#include <stdexcept>
#include <cmath>
#include <thread>  // 用于获取CPU核心数
// 获取最佳K值（std::async优化版：并行算全量结果+限制并发）
std::pair<int, std::vector<std::vector<int>>> getBestK(std::vector<Data>& data) {
	if (data.empty()) {
		throw std::invalid_argument("Data is empty in getBestK");
	}

	// 1. 预生成K候选列表（3,5,7...<√data.size()）
	int maxK = static_cast<int>(sqrt(data.size()));
	std::vector<int> kCandidates;
	for (int k = 3; k < maxK; k += 2) {
		kCandidates.push_back(k);
	}
	if (kCandidates.empty()) {
		throw std::runtime_error("No valid K candidates (data size too small)");
	}

	// 2. 关键优化：并行任务返回【准确率+混淆矩阵】，避免后续重复计算
	using TaskResult = std::tuple<double, int, std::vector<std::vector<int>>>;  // 准确率、K值、混淆矩阵
	std::vector<std::future<TaskResult>> futures;

	// 优化并发数：用CPU核心数限制最大并行线程数（避免线程爆炸）
	const int maxThreads = std::thread::hardware_concurrency();  // 自动获取CPU逻辑核心数（如8/16）
	std::atomic<int> runningTasks = 0;  // 原子变量，安全统计运行中的任务数

	for (int k : kCandidates) {
		// 控制并发：若运行中任务数达到核心数，阻塞等待
		while (runningTasks >= maxThreads) {
			std::this_thread::yield();  // 让出CPU，避免忙等
		}

		// 原子变量计数+1（线程安全）
		runningTasks++;

		// 提交异步任务：计算当前K的准确率和混淆矩阵
		futures.push_back(std::async(std::launch::async, [&data, k, &runningTasks]() {
			// 任务结束时，运行中任务数-1（确保即使抛异常也会执行）
			std::unique_ptr<int, std::function<void(int*)>> guard(nullptr, [&](int*) {
				runningTasks--;
				});

			// 一次计算拿到准确率和混淆矩阵，无需后续重复调用
			auto [acc, confusionMat] = kFoldCrossVaild(data, 10, k);
			return std::make_tuple(acc, k, confusionMat);
			}));
	}

	// 3. 收集结果，找到最佳K（准确率最高）
	double maxAcc = -1.0;
	int bestK = 3;
	std::vector<std::vector<int>> bestConfusion;

	for (auto& fut : futures) {
		// 捕获单个任务的异常（避免一个任务错导致整体崩溃）
		try {
			auto [acc, k, confusionMat] = fut.get();
			// 更新最佳K
			if (acc > maxAcc) {
				maxAcc = acc;
				bestK = k;
				bestConfusion = std::move(confusionMat);  // 移动语义，避免拷贝开销
			}
		}
		catch (const std::exception& e) {
			std::cerr << "Failed to compute K=" << e.what() << std::endl;
		}
	}

	// 4. 输出结果（若所有任务都失败，抛出异常）
	if (maxAcc < 0) {
		throw std::runtime_error("All K candidate computations failed in getBestK");
	}
	std::cout << "Best k = " << bestK << ", Accuracy = " << maxAcc << std::endl;

	return { bestK, bestConfusion };
}
#endif

// 2.计算欧几里得距离
double eucliDistance(Data x, Data y)
{
	double trail = x.length - y.length;
	double play = x.game_time - y.game_time;
	double eating = x.ice_crime_eating - y.ice_crime_eating;

	return trail * trail + play * play + eating * eating;
}

// 提取3和5的公共部分
std::vector<int> getKNeighborProb(const Data& test, const std::vector<Data>& trains, int k)
{
#ifdef OLD // 优化1.获取多类别邻居出现概率 使用优先队列(最大堆)替换sort排序降低时间开销
	if (trains.empty())
	{
		std::cout << "Null data can't be predicted!" << std::endl;
		__debugbreak();
		return {};
	}

	std::vector<std::pair<double, int>> predistance;
	for (int i = 0; i < trains.size(); ++i)
	{
		double dist = eucliDistance(trains[i], test);
		predistance.emplace_back(dist, trains[i].type);
	}

	std::sort(predistance.begin(), predistance.end());
	std::vector<int> vote(3, 0);
	for (int i = 0; i < k && i < predistance.size(); ++i)
		vote[predistance[i].second]++;
	return vote;
#else
	if (trains.empty() || k <= 0)
		throw std::invalid_argument("Invalid input for getNeighborProb!");
	std::priority_queue<std::pair<double, int>> maxHeap; // 使用最大堆存储
	for (const auto& d : trains)
	{
		double dist = eucliDistance(test, d);
		if (maxHeap.size() < k)
			maxHeap.emplace(dist, d.type);
		else if (maxHeap.top().first > dist)
		{
			maxHeap.pop();
			maxHeap.emplace(dist, d.type);
		}
	}

	std::vector<int> neighborProb(3, 0);
	while (!maxHeap.empty())
	{
		neighborProb[maxHeap.top().second]++;
		maxHeap.pop();
	}

	return neighborProb;
#endif
}

// 3.预测类型
int predicType(const Data& test, std::vector<Data>& data, int k)
{
	auto vote = getKNeighborProb(test, data, k);
	return max_element(vote.begin(), vote.end()) - vote.begin();
}

// 4.使用K折交叉验证计算准确率 结合3使用
std::pair<double, std::vector<std::vector<int>>> kFoldCrossVaild(std::vector<Data>& shuffle_data, int kfold, int knn_k)
{
	if (shuffle_data.empty())
		throw std::invalid_argument("Invalid input for countRoc: Null data is invaild!");

	if (kfold <= 0)
		throw std::invalid_argument("Invalid input for countRoc: kfold must be bigger than 0!");

	std::vector<std::vector<int>> cntlist(3, std::vector<int>(3, 0));
	int foldSize = shuffle_data.size() / kfold;
	std::vector<double> accuracies;
	for (int fold = 0; fold < kfold; ++fold)
	{
		int start = fold * foldSize;
		int end = (kfold -1 == fold) ? shuffle_data.size() : start + foldSize;
		int correct = 0;

		std::vector<Data> trainData(shuffle_data.begin(), shuffle_data.begin() + start);
		trainData.insert(trainData.end(), shuffle_data.begin() + end, shuffle_data.end());
		for(int i = start; i < end; ++i)
		{
			auto t = shuffle_data[i];
			int res = predicType(t, trainData, knn_k);
			cntlist[t.type][res]++;
			correct += (res == t.type);
		}
		double accuracy = static_cast<double>(correct) / foldSize;
		accuracies.push_back(accuracy);
	}

	return std::make_pair(std::accumulate(accuracies.begin(), accuracies.end(), 0.0) / kfold , cntlist);
}

// *****************************************************************************
// ******************************获取ROC数据************************************
// *****************************************************************************

// 5.计算测试集在训练集上的多类型概率
std::vector<double> predicProb(const Data& test, std::vector<Data>& data, int k)
{
	auto vote = getKNeighborProb(test, data, k);
	std::vector<double> appearance(3, 0);
	for (int i = 0; i < appearance.size(); ++i)
		appearance[i] = static_cast<double>(vote[i]) / k;

	return appearance;
}

// 6.使用K折交叉验证，计算ROC图数据 结合5使用
std::pair<double, double> countRoc(std::vector<Data>& shuffle_data, int kfold, int knn_k, double confd, int genre)
{
	if (shuffle_data.empty())
		throw std::invalid_argument("Invalid input for countRoc: Null data is invaild!");

	if (kfold <= 0)
		throw std::invalid_argument("Invalid input for countRoc: kfold must be bigger than 0!");

	int foldSize = shuffle_data.size() / kfold;
	int TP = 0, FP = 0, FN = 0, TN = 0;
	for (int fold = 0; fold < kfold; ++fold)
	{
		int start = fold * foldSize;
		int end = (kfold - 1 == fold) ? shuffle_data.size() : start + foldSize;
		int correct = 0;


		std::vector<Data> trainData;
		trainData.reserve(shuffle_data.size() - foldSize);
		trainData.assign(shuffle_data.begin(), shuffle_data.begin() + start);
		trainData.insert(trainData.end(), shuffle_data.begin() + end, shuffle_data.end());

		for (int i = start; i < end; ++i)
		{
			auto t = shuffle_data[i];
			std::vector<double> res = predicProb(t, trainData, knn_k);
			bool isPred = (res[genre] >= confd);
			bool isTrue = (genre == t.type);
			TP += isPred && isTrue;
			FP += isPred && (!isTrue);
			FN += (!isPred) && isTrue;
			TN += (!isPred) && (!isTrue);
		}
	}
	
	double TPR = ( TP+FN == 0 ? 0.0 : static_cast<double>(TP) / (TP + FN));
	double FPR = ( FP+TN == 0 ? 0.0 : static_cast<double>(FP) / (FP + TN));

	return { TPR, FPR };
}

// 7.组织ROC图数据并补全 结合6使用
std::pair<std::vector<double>, std::vector<double>> getROC(std::vector<Data>& data, int kfold, int knn_k, int genre)
{
	std::vector<double> tprlist;
	std::vector<double> fprlist;
	for (double confd = 1.0; confd > 0; confd -= 0.05)
	{
		auto [tpr, fpr] = countRoc(data, kfold, knn_k, confd, genre);
		tprlist.push_back(tpr);
		fprlist.push_back(fpr);
	}
	if (!(tprlist.front() == 0 && fprlist.front() == 0))
	{
		tprlist.insert(tprlist.begin(), 0);
		fprlist.insert(fprlist.begin(), 0);
	}
	if (!(tprlist.back() == 1 && fprlist.back() == 1))
	{
		tprlist.push_back(1);
		fprlist.push_back(1);
	}

	return { fprlist, tprlist };
}

// ******************************************************************************
// ******************************获取AUC数据*************************************
// ******************************************************************************

#ifdef OLD

// 9.计算一个柱子的AUC
inline double getColArea(double x1, double x2, double y1 ,double y2)
{
	return abs(x1-x2)*(y1+y2)/2;
}
#endif

// 10.累加AUC得到最终值 结合8使用 data十分有序，无需处理即可使用
double getAUC(std::pair<std::vector<double>, std::vector<double>>& data)
{
	double auc = 0.0;

	auto& fprs = data.first;
	auto& tprs = data.second;
	for (int i = 0; i < fprs.size()-1; ++i)
	{
		auc += (fprs[i + 1] - fprs[i]) * (tprs[i + 1] + tprs[i]) / 2;
	}

	std::cout << auc << std::endl;
	return auc;
}