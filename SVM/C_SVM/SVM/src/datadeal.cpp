#include "datadeal.h"

#include <numeric>

void matinfo(matvar_t* matvar)
{
	if (matvar == nullptr)
		return;
	size_t sz = 1;
	std::cout << "维度数：" << matvar->rank << " | 维度：[";
	for (size_t i = 0; i < matvar->rank; ++i)
	{
		sz *= matvar->dims[i];
		std::cout << (i == 0 ? "" : ",") << matvar->dims[i];
	}
	std::cout << "] | 数量：" << sz << std::endl;
}

std::vector<Sample> loadData(const std::string& filepath)
{
	mat_t* matfp = Mat_Open(filepath.c_str(), MAT_ACC_RDONLY);
	if (matfp == NULL)
	{
		std::cerr << "loadData: Mat file failure to open!" << std::endl;
		return {};
	}
	std::cout << "loadData: Mat file is successfully opened!" << std::endl;

	std::vector<Sample> samples;
	std::vector<std::string> vars = { "X", "y" };
	for (const std::string v : vars)
	{
		std::cout << "读取变量：" << v << std::endl;
		auto column = Mat_VarRead(matfp, v.c_str());
		if (column == nullptr)
		{
			std::cout << "失败" << std::endl;
			continue;
		}
		if (column->class_type == MAT_C_DOUBLE)
		{
			matinfo(column);
			double* matdata = (double*)column->data;

			size_t block = column->dims[0];
			if(samples.empty())
				samples.resize(block, Sample(-1, std::vector<double>(column->dims[1], 0.0)));

			for (int i = 0; i < column->dims[0]; ++i)
			{
				if (v == "X")
				{
					for (int j = 0; j < column->dims[1]; ++j)
						samples[i].features[j] = matdata[i + j * block];
				}
				else if (v == "y")
					samples[i].tag = (matdata[i] ? 1 : -1);
			}

		}
	}

	std::cout << "读取成功" << std::endl;
	Mat_Close(matfp);

	return samples;
}

void normalize(std::vector<Sample>& samples)
{
	if (samples.empty())
	{
		std::cerr << "normalize: arg is empty!" << std::endl;
		return;
	}

	const size_t sample_size = samples.size();
	const size_t feature_size = samples[0].features.size();
	std::vector<double> means(feature_size, 0);
	for (const auto& sample : samples)
		for (size_t i = 0; i < feature_size; ++i)
			means[i] += sample.features[i];

	for (auto& mean : means)
		mean /= sample_size;

	std::vector<double> deviation(feature_size, 0);
	for (const auto& sample : samples)
	{
		for (size_t i = 0; i < feature_size; ++i)
		{
			double gap = sample.features[i] - means[i];
			deviation[i] += gap * gap;
		}
	}

	for (auto& dev : deviation)
		dev = sqrt(dev/ sample_size);

	for (auto& sample : samples)
		for (size_t i = 0; i < feature_size; ++i)
			sample.features[i] = (sample.features[i] - means[i]) / deviation[i];

	std::cout << "normalize: 标准化完成" << std::endl;
}

std::unordered_map<double, std::pair<std::vector<double>, std::vector<double>>> group(const std::vector<Sample>& samples)
{
	if (samples.empty())
	{
		std::cerr << "group: arg is empty!" << std::endl;
		return {};
	}

	std::unordered_map<double, std::pair<std::vector<double>, std::vector<double>>> classifier;
	for (auto& sample : samples)
	{
		classifier[sample.tag].first.push_back(sample.features[0]);
		classifier[sample.tag].second.push_back(sample.features[1]);
	}

	return classifier;
}