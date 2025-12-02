#include "decisionTree_algorithm.h"

#include "datadeal.h"
#include "Evaluate.h"

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

template<bool ratio = false>
auto branch(std::vector<std::vector<int>>& features, std::vector<int>& tags, double(classifier)(const std::vector<int>&, int))
{
	if (features.empty())
		CatchErr("branch: 数据为空");
	int sample = tags.size();
	if (sample <= 0)
		CatchErr("branch: 数据无效");

	double base_value = (classifier == Ent ? classifier(bincount(tags), sample) : -1);
	std::vector<double> values;
	values.reserve(tags.size());

	std::vector<double> iv;
	iv.reserve(tags.size());

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
		values.push_back(base_value >= 0 ? base_value - count_value : count_value);
		iv.push_back(rate);
	}

	if constexpr (ratio)
		return std::make_pair(values, iv);
	else
		return values;
}

//std::vector<double> Gain(std::vector<Data>& data)
//{
//	auto [features, tags] = transform(data);
//	return Gain(features, tags);
//}

int Gain(std::vector<std::vector<int>>& features, std::vector<int>& tags)
{
	auto val = branch(features, tags, Ent);
	return std::max_element(val.begin(), val.end()) - val.begin();
}

int GainRatio(std::vector<std::vector<int>>& features, std::vector<int>& tags)
{
	auto [val, iv] = branch<true>(features, tags, Ent);
	if (val.empty() || iv.empty())
		return -1;

	double avg = std::accumulate(val.begin(), val.end(), 0) / val.size();
	std::vector<double> ratio;
	ratio.reserve(val.size()/2+1);
	for (int i = 0; i < val.size(); ++i)
	{
		if (val[i] > avg)
			ratio.push_back(val[i] / iv[i]);
	}

	return std::max_element(ratio.begin(), ratio.end()) - ratio.begin();
}

//std::vector<double> CART(std::vector<Data>& data)
//{
//	auto [features, tags] = transform(data);
//	return CART(features, tags);
//}

int CART(std::vector<std::vector<int>>& features, std::vector<int>& tags)
{
	auto val = branch(features, tags, Gini);
	return std::min_element(val.begin(), val.end()) - val.begin();
}

double tree_accuracy(DecisionTreeNode* root, std::vector<std::vector<int>>& features, std::vector<int>& tags)
{
	if (!tags.size())
		return 0.0;
	double passive = 0;
	std::vector<std::vector<int>> test(features[0].size(), std::vector<int>(features.size()));
	for (int i = 0; i < features.size(); ++i)
	{
		for (int j = 0; j < features[i].size(); ++j)
		{
			test[j][i] = features[i][j];
		}
	}
	for (int i = 0; i < test.size(); ++i)
		passive += predict_type(root, test[i]) == tags[i];
	return passive / tags.size();
}

double tree_accuracy(int tag, std::vector<int>& tags)
{
	double passive = 0;
	for (int t : tags)
		passive += t == tag;
	return passive / tags.size();
}

DecisionTreeNode* CreateDecisionTree(std::vector<std::vector<int>>& train_fs, std::vector<int>& train_ts,
	std::vector<int> idxlist, ClassifierFunc classifier, std::vector<bool> feature_mask,
	std::vector<std::vector<int>>& verify_fs = std::vector<std::vector<int>>(), std::vector<int>& verify_ts = std::vector<int>(),
	std::vector<bool> mask = std::vector<bool>(), bool pruning = false)
{
	std::vector<std::vector<int>> sub_features = splitlist(feature_mask, train_fs);
	std::vector<int> sub_tags = splitlist(feature_mask, train_ts);

	auto taglist = bincount(sub_tags);
	int max_tag = std::max_element(taglist.begin(), taglist.end()) - taglist.begin();

	if (unique(sub_tags).size() == 1)
		return new DecisionTreeNode(sub_tags[0], -1, max_tag, true);
	if (idxlist.size() == 0)
		return new DecisionTreeNode(max_tag, -1, max_tag, true);


	int idx = classifier(sub_features, sub_tags);
	if (idx >= sub_tags.size())
		CatchErr("CreateDecisionTree: 最大值索引超过数据集大小");
	if (idx == -1)
		CatchErr("出现空数据集，导致信息增益率计算出错");

	std::vector<int> values(sub_features[idx]);
	auto typenumlist = unique(values);

	std::vector<int> newlist(idxlist);
	newlist.erase(newlist.begin() + idx);

	std::vector<DecisionTreeNode*> children;
	children.reserve(typenumlist.size());
	for (int t : typenumlist)
	{
		std::vector<bool> new_feature_mask(feature_mask);
		for (int i = 0; i < new_feature_mask.size(); ++i)
			new_feature_mask[i] = new_feature_mask[i] && (train_fs[idx][i] == t);

		std::vector<bool> update_mask(mask);
		if(pruning)
			for(int i = 0; i < verify_fs[idx].size(); ++i)
				update_mask[i] = update_mask[i] && (verify_fs[idxlist[idx]][i] == t);
		auto child = CreateDecisionTree(train_fs, train_ts, newlist, classifier, new_feature_mask, verify_fs, verify_ts, update_mask, pruning);
		child->val = t;
		children.push_back(child);
	}

	auto tree = new DecisionTreeNode(idx, idxlist[idx], max_tag, false, children);

	if (pruning)
	{
		auto this_verify_f = splitlist(mask, verify_fs);
		auto this_verify_t = splitlist(mask, verify_ts);
		double origin = tree_accuracy(max_tag, this_verify_t);
		double modify = tree_accuracy(tree, this_verify_f, this_verify_t);
		std::cout << "--->> origin: " << origin << ", modify: " << modify << " <<---" << std::endl;

		if (origin >= modify)
			return new DecisionTreeNode(max_tag, -1, max_tag, true);
	}
	return tree;
}

DecisionTreeNode* buildDecisionTree(std::vector<std::vector<int>> train_fs, std::vector<int> train_ts,
	std::vector<int> idxlist, ClassifierFunc classifier)
{
	std::vector<bool> mask(train_ts.size(), true);

	return CreateDecisionTree(train_fs, train_ts, idxlist, classifier, mask);
}

DecisionTreeNode* PrePruning(std::vector<std::vector<int>> train_fs, std::vector<int> train_ts,
	std::vector<int> idxlist, ClassifierFunc classifier,
	std::vector<std::vector<int>>& verify_fs, std::vector<int>& verify_ts)
{
	std::vector<bool> feature_mask(train_ts.size(), true);
	std::vector<bool> mask(verify_ts.size(), true);
	return CreateDecisionTree(train_fs, train_ts, idxlist, classifier, feature_mask, verify_fs, verify_ts, mask, true);
}

DecisionTreeNode* post_pruning(DecisionTreeNode* root, std::vector<std::vector<int>>& verify_fs, std::vector<int>& verify_ts)
{
	if (root == nullptr)
		return nullptr;
	if (root->is_leaf)
		return root;
	for (int i = 0; i < root->Children.size(); ++i)
		root->Children[i] = post_pruning(root->Children[i], verify_fs, verify_ts);

	DecisionTreeNode* node = new DecisionTreeNode(root->major_type, -1, root->major_type);
	double origin = tree_accuracy(root, verify_fs, verify_ts);
	double modify = tree_accuracy(node, verify_fs, verify_ts);

	if (origin >= modify)
		return root;
	else
		return node;
}

DecisionTreeNode* PostPruning(std::vector<std::vector<int>> train_fs, std::vector<int> train_ts,
	std::vector<int> idxlist, ClassifierFunc classifier,
	std::vector<std::vector<int>>& verify_fs, std::vector<int>& verify_ts)
{
	auto tree = buildDecisionTree(train_fs, train_ts, idxlist, classifier);
	return post_pruning(tree, verify_fs, verify_ts);
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