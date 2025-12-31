#pragma once
#include <matplot\matplot.h>

#include <vector>
struct Sample;

// 打印点分布
void distrshow(const std::unordered_map<double, std::pair<std::vector<double>, std::vector<double>>>& samples);

// 打印支持向量
void supportshow(const std::unordered_map<double, std::pair<std::vector<double>, std::vector<double>>>& samples);