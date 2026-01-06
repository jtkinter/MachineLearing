#pragma once
#include <iostream>
#include <matio.h>
#include <vector>
#include <unordered_map>

struct Sample
{
	double tag;
	std::vector<double> features;

	Sample(double label = -1, std::vector<double> voc = std::vector<double>())
		:tag(label), features(voc)
	{
	}
};

struct NormalizeStats
{
	std::vector<double> means;
	std::vector<double> deviation;
};

// 读取数据
std::vector<Sample> loadData(const std::string& filepath);

// 分组方便打印
std::unordered_map<double, std::pair<std::vector<double>, std::vector<double>>> group(const std::vector<Sample>& samples);

// 标准化
NormalizeStats normalize(std::vector<Sample>& samples, const NormalizeStats& stats);

// 归一化
void uniformize(std::vector<Sample>& samples);