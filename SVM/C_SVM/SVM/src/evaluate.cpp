#include "evaluate.h"

#include "datadeal.h"
#include "svm.h"

double linear_predict(SVM& model, Sample& sample)
{
	double cnt = model.b;
	cnt += kernel(model.w, sample.features);

	double result = 0.0;
	if (fabs(cnt) > model.tol)
		result = (cnt > 0.0 ? 1.0 : -1.0);
	return result;
}

double gauss_predict(SVM& model, Sample& sample)
{
	double cnt = 0.0;
	size_t support_sz = model.support_vectors.size();
	const KernelType type = model.kernel_type;
	const double sigma = model.val;
	for (size_t i = 0; i < support_sz; ++i)
	{
		const Sample& sv = model.support_vectors[i];
		double alpha = model.alpha[i];
		double y = sv.tag;
		double val = kernel(sample.features, sv.features, type, sigma);
		cnt += alpha * y * val;
	}

	double result = 0.0;
	if (fabs(cnt) > model.tol)
		result = (cnt > 0.0 ? 1.0 : -1.0);

	return result;
}

double polynomial_predict(const SVM& model, const Sample& sample)
{
	double cnt = 0.0;
	double fx = model.b;
	size_t support_sz = model.support_vectors.size();
	for (size_t i = 0; i < support_sz; ++i)
	{
		const Sample& sv = model.support_vectors[i];
		double alpha = model.alpha[i];
		double y = sv.tag;
		double val = kernel(sv.features, sample.features, model.kernel_type, model.val);
		cnt += alpha * y * val;
	}

	double result = 0.0;
	if (fabs(cnt) > model.tol)
		result = (cnt > 0.0 ? 1.0 : -1.0);

	return result;
}

void evaluate(SVM& model, std::vector<Sample>& samples)
{
	double corrent = .0;
	for (Sample sample : samples)
	{
		double res = 0.0;
		switch (model.kernel_type)
		{
		case KernelType::LINEAR:
			res = linear_predict(model, sample);
			break;
		case KernelType::GAUSSRBF:
			res = gauss_predict(model, sample);
			break;
		case KernelType::POLYNOMIAL:
			res = polynomial_predict(model, sample);
			break;
		default:
			std::cout << "未知核技巧，使用默认的线性点乘" << std::endl;
			res = linear_predict(model, sample);
		}
		corrent += res == sample.tag;
		//std::cout << "原标签：" << sample.tag << " 预测标签：" << genre << std::endl;
	}

	std::cout << "预测的准确率：" << corrent / samples.size() << std::endl;
}