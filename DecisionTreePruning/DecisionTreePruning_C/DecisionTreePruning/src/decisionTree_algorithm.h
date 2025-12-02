#pragma once

#include <vector>
#include <cmath>
#include <queue>
#include <iostream>
#include <sstream>
#include <numeric>

struct Data;
struct DecisionTreeNode
{
	int val; // value From parent-node's feature
	int feature; // feature type
	int major_type;
	bool is_leaf;
	std::vector<DecisionTreeNode*> Children;

	DecisionTreeNode(int value, int f, int type, bool leaf = false,
		std::vector<DecisionTreeNode*> children = std::vector<DecisionTreeNode*>())
		: val(value), feature(f), major_type(type), is_leaf(leaf), Children(children)
	{
	}

	~DecisionTreeNode()
	{
		for (auto& child : Children)
			delete child;

		Children.clear();
	}

	friend std::ostream& operator<< (std::ostream& os, const DecisionTreeNode& node)
	{
		if (node.is_leaf)
			os << "result: " << node.val;
		else
			os << "feature: " << node.feature;
		return os;
	}

	std::string to_string() const
	{
		std::ostringstream oss;
		oss << *this;
		return oss.str();
	}
};


// 计算信息增益
int Gain(std::vector<std::vector<int>>& features, std::vector<int>& tags);

// 计算信息增益率
int GainRatio(std::vector<std::vector<int>>& features, std::vector<int>& tags);

// 计算基尼系数
int CART(std::vector<std::vector<int>>& features, std::vector<int>& tags);

// 创建决策树
using ClassifierFunc = int(*)(std::vector<std::vector<int>>&, std::vector<int>&);
using Strategy = DecisionTreeNode * (*)(std::vector<std::vector<int>>, std::vector<int>,
	std::vector<int> idxlist, ClassifierFunc,
	std::vector<std::vector<int>>&, std::vector<int>&);

// 创建预剪枝过的决策树
DecisionTreeNode* PrePruning(std::vector<std::vector<int>> train_fs, std::vector<int> train_ts,
	std::vector<int> idxlist, ClassifierFunc classifier,
	std::vector<std::vector<int>>& verify_fs, std::vector<int>& verify_ts);

// 后剪枝
DecisionTreeNode* PostPruning(std::vector<std::vector<int>> train_fs, std::vector<int> train_ts,
	std::vector<int> idxlist, ClassifierFunc classifier,
	std::vector<std::vector<int>>& verify_fs, std::vector<int>& verify_ts);


// 使用前序遍历打印树
void printree(DecisionTreeNode* root);

// 使用层序遍历获取树高
int getLevel(DecisionTreeNode* root);