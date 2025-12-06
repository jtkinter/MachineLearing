#include "datadeal.h"
#include "bayes.h"

int main()
{
	auto dataset = loadData("source/encoded_dataset.txt");
	auto testset = loadData("source/encoded_testset.txt");

	auto compressed = compressData(dataset);

	auto prior_prob_list = compressed.first;
	auto ds_list = compressed.second.first;
	auto ct_list = compressed.second.second;
	
	auto res = predict(prior_prob_list, ds_list, ct_list, testset);
	for (int r : res)
		cout << "预测结果：" << r << endl;

	//cout << "先验概率" << endl;
	//for (auto& prior_prob : compressed.first)
	//	cout << prior_prob.first << " -> " << prior_prob.second << endl;

	//cout << "离散值概率" << endl;
	//for (auto& ds_list : compressed.second.first)
	//{
	//	int i = 0;
	//	for (auto& maps : ds_list)
	//	{
	//		cout << i++ << endl;
	//		for (auto& port : maps)
	//			cout << port.first << " -> " << port.second << endl;
	//	}
	//}

	//cout << "连续值概率密度" << endl;
	//int i = 0;
	//for (auto& ct_list : compressed.second.second)
	//{
	//	cout << i++ << endl;
	//	for (auto& val : ct_list)
	//		cout << val.mean << " " << val.variance << " " << val.deviation << endl;
	//}

	return 0;
}