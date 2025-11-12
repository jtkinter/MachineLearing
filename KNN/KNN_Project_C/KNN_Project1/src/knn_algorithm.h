#pragma once

#include <iostream>
#include <vector>

struct Data;

std::pair<int, std::vector<std::vector<int>>> getBestK(std::vector<Data>& data);
std::pair<double, std::vector<std::vector<int>>> kFoldCrossVaild(std::vector<Data>& shuffle_data, int kfold, int knn_k);
std::pair<double, double> countRoc(std::vector<Data>& data, int kfold, int knn_k, double confd, int genre);
std::pair<std::vector<double>, std::vector<double>> getROC(std::vector<Data>& data, int kfold, int knn_k, int genre);
double getAUC(std::pair<std::vector<double>, std::vector<double>>& data);