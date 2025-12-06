#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <algorithm>
using namespace std;

struct Data
{
	vector<int> disperse;
	vector<double> continuous;
	int tag;
};

// 异常处理
#define CatchErr(msg) do{\
	std::cerr << "\033[31m"<< msg << "\033[30m" << std::endl;\
	throw std::invalid_argument(msg);\
} while(0);

// 断言
#define Assert(condition, msg) if(!condition) CatchErr(msg)

// 导入数据
vector<Data> loadData(const string& filepath);

// 转置
pair<vector<vector<int>>, vector<vector<double>>> transform(const vector<Data>& data);


// 统计出现类型及其次数，返回列表
template<bool return_counts = false, class T>
auto unique(const vector<T>& buffer, bool order = true)
{
	unordered_map<T, int> map;
	for (T type : buffer)
		map[type]++;

	vector<T> genres;
	genres.reserve(map.size());
	for (const auto& m : map)
		genres.push_back(m.first);
	if (order)
		sort(genres.begin(), genres.end());

	if constexpr (return_counts)
	{
		vector<T> counts;
		counts.reserve(genres.size());
		for (T val : genres)
			counts.push_back(map[val]);
		return make_pair(genres, counts);
	}
	else
		return genres;
}

// 计算类型出现次数
template<class T>
vector<int> bincount(const vector<T>& buffer)
{
	return unique<true, T>(buffer).second;
}