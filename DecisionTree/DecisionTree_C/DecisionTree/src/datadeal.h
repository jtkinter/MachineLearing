#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>

struct Data
{
	std::vector<int> features;
	int tag;
};

// 异常处理
#define CatchErr(msg) do{\
	std::cerr << "\033[31m"<< msg << "\033[30m" << std::endl;\
	throw std::invalid_argument(msg);\
} while(0);

// 导入数据
std::vector<Data> loadData(const std::string filepath);

// 打印数据
void printData(std::vector<Data>& dataSet);

// 转置数据
std::pair<std::vector<std::vector<int>>, std::vector<int>> transform(const std::vector<Data>& data);

// 仅返回次数列表
std::vector<int> bincount(const std::vector<int>& buffer);

// 生成对应值的布尔列表
std::vector<bool> in(const std::vector<int>& buffer, int element);

// 根据布尔列表生成对应的列表
std::vector<int> splitlist(const std::vector<bool>& idxlist, const std::vector<int>& source);
std::vector<std::vector<int>> splitlist(const std::vector<bool>& idxlist, std::vector<std::vector<int>> source);

// 统计出现类型及其次数，返回列表
template<bool return_counts = false>
auto unique(const std::vector<int>& buffer, bool sort = true)
{
	std::unordered_map<int, int> map;
	for (int type : buffer)
		map[type]++;

	std::vector<int> genres;
	genres.reserve(map.size());
	for (const auto& m : map)
		genres.push_back(m.first);
	if (sort)
		std::sort(genres.begin(), genres.end());

	if constexpr (return_counts)
	{
		std::vector<int> counts;
		counts.reserve(genres.size());
		for (int val : genres)
			counts.push_back(map[val]);
		return std::make_pair(genres, counts);
	}
	else
		return genres;
}