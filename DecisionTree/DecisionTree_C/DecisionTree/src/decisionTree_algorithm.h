#pragma once

#include <vector>
#include <cmath>
#include <queue>
#include <iostream>
#include <sstream>

struct Data;
struct DecisionTreeNode
{
	int val; // feature type
	int feature; // From parent-node's feature
	bool is_leaf;
	std::vector<DecisionTreeNode*> Children;

	DecisionTreeNode(int value, int f, bool leaf = false,
		std::vector<DecisionTreeNode*> children = std::vector<DecisionTreeNode*>())
		: val(value), feature(f), is_leaf(leaf), Children(children)
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
std::vector<double> Gain(std::vector<std::vector<int>>& features, std::vector<int>& tags);

// 计算信息增益率
std::vector<double> GainRatio(std::vector<std::vector<int>>& features, std::vector<int>& tags);

// 计算基尼系数
std::vector<double> CART(std::vector<std::vector<int>>& features, std::vector<int>& tags);

// 创建决策树
using ClassifierFunc = std::vector<double>(*)(std::vector<std::vector<int>>&, std::vector<int>&);
DecisionTreeNode* CreateDecistionTree(std::vector<std::vector<int>> features, std::vector<int> tags,
	std::vector<int> idxlist, ClassifierFunc classifier);

// 使用前序遍历打印树
void printree(DecisionTreeNode* root);

// 使用层序遍历获取树高
int getLevel(DecisionTreeNode* root);