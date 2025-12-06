#include "bayes.h"

#include "datadeal.h"
#include <numeric>

#define RAW

double prob_density(ContinuousStats& dataset, double test)
{
	double pi = 4 * atan(1.0);
	if (dataset.variance < 1e-8) dataset.variance = 1e-8;
	if (dataset.deviation < 1e-8) dataset.deviation = sqrt(dataset.variance);

	double gap = test - dataset.mean;

#ifndef RAW
	
	double den = -0.5 * log(2 * pi) - log(dataset.deviation);
	double mol = -gap * gap / (2 * dataset.variance);
	return den+mol;

#else

	double res = (1 / (sqrt(2 * pi) * dataset.deviation)) * exp(-gap * gap / (2 * dataset.variance));
	return res;

#endif

}

ContinuousStats stats(std::vector<double>& values)
{
	Assert(!values.empty(), "stats: values is empty!");
	double avg = accumulate(values.begin(), values.end(), 0.0) / values.size();
	double sum = 0.0;
	for (double value : values)
	{
		double gap = value - avg;
		sum += gap * gap;
	}
	double variance = sum / values.size();
	double deviation = sqrt(variance);

	return {avg, variance, deviation};
}

pair<
	unordered_map<int, double>, 
	pair<
		vector<vector<unordered_map<int, double>>>, 
		vector<vector<ContinuousStats>>
		>
	> compressData(vector<Data>& data)
{
	Assert(!data.empty(), "compressData: data is empty");

	vector<vector<unordered_map<int, double>>> ds_list;
	vector<vector<ContinuousStats>> ct_list;

	auto transformation = transform(data);
	vector<vector<int>> ds_tf(transformation.first.begin(), transformation.first.end() - 1);
	vector<int> tag_tf(*(transformation.first.end() - 1));
	vector<vector<double>> ct_tf = transformation.second;

	double total = data.size();
	int ds_num = data[0].disperse.size();
	int ct_num = data[0].continuous.size();

	unordered_map<int, int> group_data;
	for (int tag : tag_tf)
		group_data[tag]++;

	ds_list.resize(group_data.size());
	ct_list.resize(group_data.size());

	vector<int> ds_vocab;
	ds_vocab.reserve(ds_num);
	for (auto& ds : ds_tf)
		ds_vocab.push_back(unique(ds).size());
		
	unordered_map<int, double> prior_prob;
	prior_prob.reserve(group_data.size());

	int tag_pos = 0;
	for (const auto& d : group_data)
	{
		// 计算先验概率
	#ifndef RAW
		prior_prob[d.first] = log(d.second / total);
	#else
		prior_prob[d.first] = d.second / total;
	#endif
		// 计算离散特征的概率
		cout << "离散特征：" << endl;
		vector<unordered_map<int, double>> ds_prob;
		ds_prob.reserve(ds_num);
		for (int i = 0; i < ds_num; ++i)
		{
			auto ds = ds_tf[i];
			int vocab = ds_vocab[i];
			unordered_map<int, double> feature_prob;
			for (int j = 0; j < total; ++j)
			{
				if(d.first == tag_tf[j])
					feature_prob[ds[j]]++;
			}
			for (auto& port : feature_prob)
			{
				port.second = (port.second + 1) / (d.second + vocab);
	#ifndef RAW
				port.second = log(port.second);
	#endif
				cout << port.second << " ";
			}
			cout << endl;
			ds_prob.push_back(feature_prob);
		}
		ds_list[tag_pos] = move(ds_prob);

		// 计算连续特征的概率密度
		cout << "连续特征：" << endl;
		vector<ContinuousStats> feature_density;
		feature_density.reserve(ct_num);
		for (auto& ct : ct_tf)
		{
			vector<double> sub;
			for (int j = 0; j < total; ++j)
			{
				if (d.first == tag_tf[j])
					sub.push_back(ct[j]);
			}
			feature_density.push_back(stats(sub));
		}
		for (auto& feature : feature_density)
			cout << feature.mean << " " << feature.variance << " " << feature.deviation << endl;
		ct_list[tag_pos] = move(feature_density);

		tag_pos++;
	}

	return make_pair(prior_prob, make_pair(ds_list, ct_list));
}

int bayes(unordered_map<int, double> prior_prob,
	vector<vector<unordered_map<int, double>>> ds_list,
	vector<vector<ContinuousStats>> ct_list, Data test)
{
	double max = -1e18;
	int label = -1;
	for (const auto& port : prior_prob)
	{
		double val = port.second;

#ifndef RAW
		for (int i = 0; i < test.disperse.size(); ++i)
			val += ds_list[port.first][i][test.disperse[i]];
		for (int i = 0; i < test.continuous.size(); ++i)
			val += prob_density(ct_list[port.first][i], test.continuous[i]);
#else
		for (int i = 0; i < test.disperse.size(); ++i)
			val *= ds_list[port.first][i][test.disperse[i]];
		for (int i = 0; i < test.continuous.size(); ++i)
			val *= prob_density(ct_list[port.first][i], test.continuous[i]);
#endif
		cout << val << endl;
		if (max < val)
		{
			max = val;
			label = port.first;
		}
	}

	return label;
}

vector<int> predict(unordered_map<int, double> prior_prob,
	vector<vector<unordered_map<int, double>>> ds_list,
	vector<vector<ContinuousStats>> ct_list, vector<Data>& testset)
{
	vector<int> results;
	results.reserve(testset.size());
	for (auto& test : testset)
	{
		int res = bayes(prior_prob, ds_list, ct_list, test);
		results.push_back(res);
	}

	return results;
}