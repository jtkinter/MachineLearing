#pragma once
#include <vector>
#include <iostream>

enum class KernelType
{
	LINEAR,		// 线性
	GAUSSRBF,	// 高斯RBF核
	POLYNOMIAL	// 多项式核
};

struct Sample;
struct Model
{
	// 训练样本
	std::vector<Sample> samples;
	std::vector<double> errors;

	// 超参数
	KernelType kernel_type;
	const double tolerant; // 容忍度
	const size_t max_iteration; // 迭代次数
	const size_t max_continue_no_iter; // 最大连续未修改迭代次数
	double C; // 惩罚参数
	double sigma; // 高斯RBF核相关超参数
	double poly_order; // 多项式核相关超参数

	// 超平面参数
	double b;
	std::vector<double> w, alpha;

	Model(std::vector<Sample> data = {}, KernelType type = KernelType::LINEAR,
		double tol = 1e-6, size_t max_it = 15, size_t it = 50, double c = 1.0, double sigma_val = 1.0, double order = 3.0);

	// 获取核相关参数
	double get_kernel_val() const;
};

struct SVM
{
	// 超平面参数
	double b, valid_value, tol;
	KernelType kernel_type;
	std::vector<double> w, alpha;
	std::vector<Sample> support_vectors;
	double val; // 核相关参数：高斯RBF核->sigma，多项式核->order

	SVM(const Model& model, double valid = 1e-8);
};

// 核技巧--包括线性的内积
double kernel(const std::vector<double>& x, const std::vector<double>& y, const KernelType = KernelType::LINEAR, const double val = 1.0);

// 建立模型
Model build_model(std::vector<Sample> samples, const KernelType = KernelType::LINEAR);