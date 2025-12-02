#include "Evaluate.h"

#include "decisionTree_algorithm.h"
#include "datadeal.h"

int predict_type(DecisionTreeNode* root, std::vector<int>& test)
{
	if (root == nullptr)
		return -1;
	if (root->is_leaf)
		return root->val;
	int feature = test[root->feature];
	for (auto& child : root->Children)
		if (child->val == feature)
			return predict_type(child, test);
	return -1;
}

double Accurancy(DecisionTreeNode* root, std::vector<Data> features)
{
	if (root == nullptr)
		return 0.0;
	int correct = 0;
	for (auto& data : features)
	{
		int res = predict_type(root, data.features);
		correct += res == data.tag;
	}
	return static_cast<double>(correct) / features.size();
}

double Accurancy(const std::vector<std::vector<int>>& cfmt)
{
	if (cfmt.size() == 0) return 0.0;

	int sz = cfmt.size();
	int correct = 0;
	int total = 0;
	for (int i = 0; i < sz; ++i)
	{
		if (cfmt[i].size() != sz)
			return 0.0;
		correct += cfmt[i][i];
		total += std::accumulate(cfmt[i].begin(), cfmt[i].end(), 0);
	}

	return total ? static_cast<double>(correct)/total : 0.0;
}

std::pair < std::vector<std::vector<int>>, std::unordered_map<int, int>>
ConfusionMatrix(DecisionTreeNode* root, std::vector<Data>& datas, std::vector<int>& tags)
{
	std::vector<int> classlist = unique(tags);
	std::unordered_map<int, int> tag_map, reverse_map;
	tag_map.reserve(classlist.size());
	for (int i = 0; i < classlist.size(); ++i)
	{
		tag_map.insert({ classlist[i], i });
		reverse_map.insert({ i, classlist[i] });
	}
	std::vector<std::vector<int>> cfMat(classlist.size(), std::vector<int>(classlist.size(), 0));

	for (int i = 0; i < datas.size(); ++i)
	{
		int res = predict_type(root, datas[i].features);
		if (find(classlist.begin(), classlist.end(), res) != classlist.end())
			cfMat[tag_map[tags[i]]][tag_map[res]]++;
	}

	return std::make_pair(cfMat, reverse_map);
}

std::unordered_map<int, std::pair<double, double>> Evaluate(std::vector<std::vector<int>>& cfmt, std::unordered_map<int, int>& reverse_map)
{
	std::unordered_map<int, std::pair<double, double>> evaluation;
	for (int i = 0; i < cfmt.size(); ++i)
	{
		int tp = cfmt[i][i];
		double rd = std::accumulate(cfmt[i].begin(), cfmt[i].end(), 0);
		double pd = 0;
		for (int j = 0; j < cfmt.size(); ++j)
		{
			pd += cfmt[j][i];
		}
		double precision = pd ? tp / pd : 0.0;
		double recall = rd ? tp / rd : 0.0;
		evaluation.insert({ reverse_map[i], std::make_pair(precision, recall) });
	}

	return evaluation;
}