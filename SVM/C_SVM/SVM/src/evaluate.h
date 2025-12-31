#pragma once

#include<vector>

struct Sample;
struct SVM;

void evaluate(SVM& model, std::vector<Sample>& samples);