#pragma once

#include "datadeal.h"
//
//struct Data;

struct ContinuousStats
{
	double mean; // 平均值
	double variance; // 方差
	double deviation; // 标准差
};

// 生成概率表
pair<
	unordered_map<int, double>,
	pair<
	vector<vector<unordered_map<int, double>>>,
	vector<vector<ContinuousStats>>
	>
> compressData(vector<Data>& data);

// 预测标签
vector<int> predict(unordered_map<int, double> prior_prob,
	vector<vector<unordered_map<int, double>>> ds_list,
	vector<vector<ContinuousStats>> ct_list, vector<Data>& testset);