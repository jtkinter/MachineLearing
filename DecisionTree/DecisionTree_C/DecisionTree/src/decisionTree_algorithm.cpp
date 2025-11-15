#include "decisionTree_algorithm.h"

#include "datadeal.h"

// 计算熵
double Ent(const std::vector<int>& fqcy, int num)
{
	if (num <= 0)
		CatchErr("Ent: 除零错误");
	double sum = 0.0;
	for (auto& f : fqcy)
	{
		if (f == 0) continue;
		double prob = double(f) / num;
		sum += prob*log(prob)/log(2);
	}
	return -sum;
}

// 计算离散的洛伦兹曲线
double Gini(const std::vector<int>& fqcy, int num)
{
	if (num <= 0)
		CatchErr("Gini: 除零错误");
	double sum = 1.0;
	for (auto& f : fqcy)
	{
		if (f == 0) continue;
		double prob = double(f) / num;
		sum -= prob*prob;
	}
	return sum;
}

std::vector<double> branch(std::vector<std::vector<int>>& features, std::vector<int>& tags, double(classifier)(const std::vector<int>&, int), bool ratio = false)
{
	if (features.empty())
		CatchErr("branch: 数据为空");
	int sample = tags.size();
	if (sample <= 0)
		CatchErr("branch: 数据无效");

	double base_value = (classifier == Ent ? classifier(bincount(tags), sample) : -1);
	std::vector<double> values;
	values.reserve(tags.size());
	for (auto& feature : features)
	{
		auto& [tag, cnt] = unique<true>(feature);
		int sz = tag.size();
		double rate = ratio ? classifier(cnt, sample) : 1;
		double count_value = 0.0;
		for (int i = 0; i < sz; ++i)
		{
			int t = tag[i];
			int c = cnt[i];
			auto sublist = splitlist(in(feature, t), tags);
			count_value += ((double)c / sample) * classifier(bincount(sublist), c);
		}
		values.push_back(base_value >= 0 ? (base_value - count_value)/rate : count_value);
	}

	return values;
}

//std::vector<double> Gain(std::vector<Data>& data)
//{
//	auto [features, tags] = transform(data);
//	return Gain(features, tags);
//}

std::vector<double> Gain(std::vector<std::vector<int>>& features, std::vector<int>& tags)
{
	return branch(features, tags, Ent);
}

std::vector<double> GainRatio(std::vector<std::vector<int>>& features, std::vector<int>& tags)
{
	return branch(features, tags, Ent, true);
}

//std::vector<double> CART(std::vector<Data>& data)
//{
//	auto [features, tags] = transform(data);
//	return CART(features, tags);
//}

std::vector<double> CART(std::vector<std::vector<int>>& features, std::vector<int>& tags)
{
	return branch(features, tags, Gini);
}

DecisionTreeNode* CreateDecistionTree(std::vector<std::vector<int>> features, std::vector<int> tags,
	std::vector<int> idxlist, ClassifierFunc classifier)
{
	if (unique(tags).size() == 1)
		return new DecisionTreeNode(tags[0], -1, true);
	if (idxlist.size() == 0)
	{
		auto taglist = bincount(tags);
		int max_tag = std::max_element(taglist.begin(), taglist.end()) - taglist.begin();
		return new DecisionTreeNode(max_tag, -1, true);
	}

	auto val = classifier(features, tags);
	int idx = (classifier == Gain || classifier == GainRatio ? std::max_element(val.begin(), val.end()) : std::min_element(val.begin(), val.end())) - val.begin();
	if (idx >= features.size())
		CatchErr("CreateDecisionTree: 最大值索引超过数据集大小");

	std::vector<int> values(features[idx]);
	auto typenumlist = unique(values);

	features.erase(features.begin() + idx);
	std::vector<int> newlist(idxlist);
	newlist.erase(newlist.begin() + idx);

	std::vector<DecisionTreeNode*> children;
	children.reserve(values.size());
	for (int t : typenumlist)
	{
		std::vector<bool> sublist = in(values, t);
		auto sub_features = splitlist(sublist, features);
		auto sub_tags = splitlist(sublist, tags);
		auto child = CreateDecistionTree(sub_features, sub_tags, newlist, classifier);
		child->val = t;
		children.push_back(child);
	}

	return new DecisionTreeNode(idx, idxlist[idx], false, children);
}

// 前序遍历
void printree(DecisionTreeNode* root)
{
	if (root == nullptr)
		return;
	if (root->is_leaf)
	{
		std::cout << "**" << root->val;
		return;
	}
	else
		std::cout << root->feature << "(";

	for (auto& child : root->Children)
	{
		printree(child);
		std::cout << ",";
	}

	std::cout << ")";
}

// 层序遍历
int getLevel(DecisionTreeNode* root)
{
	std::queue<DecisionTreeNode*> tree_queue;
	tree_queue.push(root);
	int level = 0;
	while (!tree_queue.empty())
	{
		int sz = tree_queue.size();
		level++;
		while (sz--)
		{
			DecisionTreeNode* node = tree_queue.front();
			tree_queue.pop();
			for (auto& child : node->Children)
			{
				tree_queue.push(child);
			}
		}
	}
	return level;
}