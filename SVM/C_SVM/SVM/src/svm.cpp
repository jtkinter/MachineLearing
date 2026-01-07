#include "svm.h"

#include <random>

#include "datadeal.h"

Model::Model(std::vector<Sample> data, KernelType type, double tol,
	size_t max_it, size_t it, double c, double sigma_val, double order)
	: samples(std::move(data)), kernel_type(type), tolerant(tol),
	max_continue_no_iter(max_it), max_iteration(it), C(c), sigma(sigma_val), poly_order(order), b(0.0)
{

	if (this->samples.empty())
	{
		std::cerr << "Model: 输入样本集为空！" << std::endl;
		return;
	}

	alpha.resize(this->samples.size(), 0);
	w.resize(this->samples[0].features.size(), 0);
	errors.resize(this->samples.size(), 0);

	if (this->C <= 0.0) {
		std::cerr << "Model: 惩罚参数C必须大于0，已修正为1.0！" << std::endl;
		this->C = 1.0;
	}
}

double Model::get_kernel_val() const
{
	switch (this->kernel_type)
	{
	case KernelType::LINEAR:
		return 1.0;
	case KernelType::GAUSSRBF:
		return this->sigma;
	case KernelType::POLYNOMIAL:
		return this->poly_order;
	default:
		std::cout << "SVM: 未知核技巧，使用1.0为默认核相关参数" << std::endl;
		return 1.0;
	}
}

SVM::SVM(const Model& model, double valid)
	:b(model.b), w(model.w), valid_value(valid), kernel_type(model.kernel_type), tol(model.tolerant)
{
	for (size_t i = 0; i < model.alpha.size(); ++i)
	{
		if (model.alpha[i] > valid_value)
		{
			this->alpha.push_back(model.alpha[i]);
			this->support_vectors.push_back(model.samples[i]);
		}
	}

	this->val = model.get_kernel_val();
}

// 向量点积
double dot(const std::vector<double>& x, const std::vector<double>& y)
{
	if (x.empty() || y.empty())
	{
		std::cerr << "dot: arg is empty!" << std::endl;
		return 0.0;
	}
	if (x.size() != y.size())
	{
		std::cerr << "dot: 向量不等长" << std::endl;
		return 0.0;
	}

	double sum = 0.0;
	for (size_t i = 0; i < x.size(); ++i)
		sum += x[i] * y[i];
	return sum;
}

// 高斯RBF核
double gaussRBF(const std::vector<double>& x, const std::vector<double>& y, const double sigma)
{
	if (x.empty() || y.empty())
	{
		std::cerr << "gaussRBF: 参数为空！" << std::endl;
		return 0.0;
	}
	if (x.size() != y.size())
	{
		std::cerr << "gaussRBF: 向量不等长" << std::endl;
		return 0.0;
	}

	double sum = 0.0;
	for (size_t i = 0; i < x.size(); ++i)
	{
		double diff = x[i] - y[i];
		sum += diff * diff;
	}

	return exp(-sum / (2 * sigma * sigma));
}

// 多项式核
double polynomial(const std::vector<double>& x, const std::vector<double>& y, const double order)
{
	return std::pow(dot(x, y) + 1.0, order);
}

// 核技巧
double kernel(const std::vector<double>& x, const std::vector<double>& y, const KernelType kernel_type, const double val)
{
	switch (kernel_type)
	{
	case KernelType::LINEAR:
		return dot(x, y);
	case KernelType::GAUSSRBF:
		return gaussRBF(x, y, val);
	case KernelType::POLYNOMIAL:
		return polynomial(x, y, val);
	default:
		std::cout << "没有找到对应的核技巧类型，使用默认的线性处理" << std::endl;
		return dot(x, y);
	}
}

// 计算误差
double calculate_error(const Model& model, size_t idx)
{
	if (model.samples.empty() || idx >= model.samples.size())
	{
		std::cerr << "calculate_error: invaild samples index!" << std::endl;
		return 0.0;
	}

	if (model.samples[0].features.empty() || model.alpha.size() != model.samples.size()) {
		std::cerr << "calculate_error: 模型参数未正确初始化！" << std::endl;
		return 0.0;
	}

	double fx = model.b;
	const Sample& sample = model.samples[idx];
	for (size_t i = 0; i < model.samples.size(); ++i)
	{
		const Sample& sk = model.samples[i];
		fx += model.alpha[i] * sk.tag * kernel(sk.features, sample.features,
			model.kernel_type, model.get_kernel_val());
	}

	return fx - sample.tag;
}

// 启发式策略：寻找j
size_t find_other(Model& model, size_t idx, double e1)
{
	if (model.samples.size() <= 1)
	{
		std::cerr << "find_other: 样本数量不足，无法选择成对样本！" << std::endl;
		return idx;
	}
	if (idx >= model.samples.size())
	{
		std::cerr << "find_other: 样本索引无效！" << std::endl;
		return 0;
	}

	size_t best_j = idx;
	double max_diff = 0.0;

	for (size_t i = 0; i < model.samples.size(); ++i)
	{
		if (i == idx) continue;
		double e2 = model.errors[i];
		double diff = fabs(e1 - e2);
		if (diff > max_diff)
		{
			max_diff = diff;
			best_j = i;
		}
	}

	if (best_j == idx)
	{
		std::cout << "find_other: 没找到合适的alpha2，已经随机指定一个" << std::endl;

		std::random_device rd;
		std::mt19937 gen(rd());

		std::uniform_int_distribution<> dist(0, model.samples.size() - 1);
		do
		{
			best_j = dist(gen);
		} while (best_j == idx);
	}

	return best_j;
}

// 裁剪alpha_j，使值落入l~h中
double clip(double l, double h, double x)
{
	if (x < l) return l;
	if (x > h) return h;
	return x;
}

// 更新维度权重
void update_w_final(Model& model) 
{
	if (model.kernel_type != KernelType::LINEAR)
		return;

	if (model.samples.empty() || model.samples[0].features.empty())
	{
		std::cerr << "update_w_final: 模型参数无效，无法更新w！" << std::endl;
		return;
	}

	std::fill(model.w.begin(), model.w.end(), 0.0);
	for (size_t i = 0; i < model.samples.size(); ++i) {
		const Sample& sample = model.samples[i];
		for(size_t j = 0; j < sample.features.size(); ++j)
			model.w[j] += model.alpha[i] * sample.tag * sample.features[j];
	}
}

// 更新误差向量
void calculate_error_model(Model& model)
{
	for (size_t i = 0; i < model.alpha.size(); ++i)
	{
		model.errors[i] =  calculate_error(model, i);
	}
}

// 使用SMO算法训练
void train(Model& model)
{
	if (model.samples.empty())
	{
		std::cerr << "train: 样本为空，无法训练！" << std::endl;
		return;
	}

	int it = 0;
	int continue_no_update = 0;
	calculate_error_model(model);
	const KernelType type = model.kernel_type;
	const double val = model.get_kernel_val();
	while (it < model.max_iteration && continue_no_update < model.max_continue_no_iter)
	{
		int change_alpha = 0;
		for (size_t i = 0; i < model.samples.size(); ++i)
		{
			double e1 = model.errors[i];
			double a1 = model.alpha[i];
			double y1 = model.samples[i].tag;
			double fx1 = e1 + y1;
			double cases = y1 * fx1;

			bool kkt_case = (a1 > model.tolerant && a1 < model.C - model.tolerant && fabs(cases)-1.0 > model.tolerant)
				|| (a1 <= model.tolerant && cases < 1.0 - model.tolerant)
				|| (a1 >= model.C - model.tolerant && cases > 1.0 + model.tolerant);
			if (!kkt_case) continue;

			size_t j = find_other(model, i, e1);

			Sample& sp1 = model.samples[i];
			Sample& sp2 = model.samples[j];

			double e2 = model.errors[j];
			double a2 = model.alpha[j];
			double y2 = sp2.tag;

			double l, h;
			if (y1 == y2)
			{
				l = std::max(0.0, a2 + a1 - model.C);
				h = std::min(model.C, a1 + a2);
			}
			else
			{
				l = std::max(0.0, a2 - a1);
				h = std::min(model.C, model.C + a2 - a1);
			}
			if (fabs(l - h) < model.tolerant)
			{
				//std::cout << "train: " 
				//	<< "当前alpha组: " << "("<< i << "," << j << ") "
				//	<< "上下界几乎一致，给" << i << "设置冷静期" << std::endl;
				continue;
			}

			double k11 = kernel(sp1.features, sp1.features, type, val);
			double k12 = kernel(sp1.features, sp2.features, type, val);
			double k22 = kernel(sp2.features, sp2.features, type, val);

			double eta = k11 + k22 - 2 * k12;
			if (eta <= model.tolerant)
			{
				//std::cout << "train: eta过小，给" << i << "设置冷静期" << std::endl;
				continue;
			}

			model.alpha[j] += y2 * (e1 - e2) / eta;
			model.alpha[j] = clip(l, h, model.alpha[j]);
			if (fabs(model.alpha[j] - a2) < model.tolerant)
			{
				//std::cout << "train: " << "当前alpha组" << "(" << i << "," << j << ") alpha[j]变化过小，已跳过" << std::endl;
				continue;
			}
			model.alpha[i] += y1 * y2 * (a2 - model.alpha[j]);

			double b1 = model.b - e1 - y1 * (model.alpha[i] - a1) * k11 - y2 * (model.alpha[j] - a2) * k12;
			double b2 = model.b - e2 - y2 * (model.alpha[j] - a2) * k22 - y1 * (model.alpha[i] - a1) * k12;

			if (model.alpha[i] > model.tolerant && model.alpha[i] < model.C - model.tolerant) model.b = b1;
			else if (model.alpha[j] > model.tolerant && model.alpha[j] < model.C - model.tolerant) model.b = b2;
			else model.b = (b1 + b2) / 2;

			std::cout << "(" << i << ", " << j << ")  "
				<< "alpha1: " << model.alpha[i] << "  alpha2: " << model.alpha[j]
				<< "  b = " << model.b << std::endl;

			model.errors[i] = calculate_error(model, i);
			model.errors[j] = calculate_error(model, j);

			change_alpha++;
		}

		it++;
		if (change_alpha == 0)
		{
			continue_no_update++;
			std::cout << "第" << it << "次无alpha值更新，连续无更新" << continue_no_update << "次..." << std::endl;
		}
		else
		{
			if (model.kernel_type == KernelType::LINEAR)
			{
				update_w_final(model);
				std::cout << "w: ";
				for (double w : model.w)
					std::cout << w << " ";
				std::cout << std::endl;
			}
			continue_no_update = 0;
			std::cout << "第" << it << "次更新了" << change_alpha*2 << "个alpha值..." << std::endl;
		}
	}
}

Model build_model(std::vector<Sample> samples, const KernelType type)
{
	Model model(samples, type);
	if (samples.empty())
	{
		std::cerr << "SMO: arg is empty!" << std::endl;
		return model;
	}

	train(model);
	std::cout << "build_model: 模型训练成功！" << std::endl;

	return model;
}