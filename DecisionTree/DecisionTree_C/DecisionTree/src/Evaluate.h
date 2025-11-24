#pragma once

#include <vector>
#include <unordered_map>


struct DecisionTreeNode;
struct Data;

// 获取模型准确率
double Accurancy(DecisionTreeNode* root, std::vector<Data> features);
double Accurancy(const std::vector<std::vector<int>>& cfmt);

// 获取混淆矩阵
std::pair < std::vector<std::vector<int>>, std::unordered_map<int, int>>
ConfusionMatrix(DecisionTreeNode* root, std::vector<Data>& datas, std::vector<int>& tags);

// 通过精确率和召回率评估模型
std::unordered_map<int, std::pair<double, double>> Evaluate(std::vector<std::vector<int>>& cfmt, std::unordered_map<int, int>& reverse_map);