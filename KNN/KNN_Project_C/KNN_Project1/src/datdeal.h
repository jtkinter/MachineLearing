#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <unordered_map>

//using namespace std; // 不能使用，在matplot++中也有std作用域，会冲突

//#define OLD

struct Data
{
	// 为什么不用vector<double>，KNN算法需要频繁访问内存的，使用vector管理反而更慢
	double length;
	double game_time;
	double ice_crime_eating;
	int type;
};

std::vector<Data> loadData(const std::string& filepath);
void normalized(std::vector<Data>& data);
void saveData(std::vector<Data>& data, const std::string& savepath);

std::vector<Data> shuffleData(std::vector<Data>& data);