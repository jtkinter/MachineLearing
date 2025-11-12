#include "datdeal.h"
#include "knn_algorithm.h"
#include "matplot_show.h"

int main()
{
	auto data = loadData("source\\datingTestSet.txt");
	normalized(data);
	auto shuffle_Data = shuffleData(data);
	auto [best_k, list] = getBestK(data);

	std::vector<std::pair<std::vector<double>, std::vector<double>>> results;
	std::vector<double> AUCs;
	for (int i = 0; i < 3; ++i)
	{
		auto res = getROC(shuffle_Data, 10, best_k, i);
		auto auc = getAUC(res);
		results.push_back(res);
		AUCs.push_back(auc);
	}
	plotEvalution(results, AUCs, list);

	return 0;
}