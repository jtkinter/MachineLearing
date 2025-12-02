#pragma once
#include <matplot/matplot.h>

struct DecisionTreeNode;

// 决策树可视化
void plotDecisionTree(DecisionTreeNode* root, matplot::axes_handle ax);

// 热力图可视化
void plotHeatmap(std::vector<std::vector<int>>& confunsion, std::unordered_map<int, int> reverse_map, matplot::axes_handle ax);