#pragma once

#include <matplot/matplot.h>

void plotROC(std::vector<std::pair<std::vector<double>, std::vector<double>>> data);
void plotEvalution(std::vector<std::pair<std::vector<double>, std::vector<double>>> data, std::vector<double> aucs, std::vector<std::vector<int>>& list);
//void plotList(std::vector<std::vector<int>>& data);