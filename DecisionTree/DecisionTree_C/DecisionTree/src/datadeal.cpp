#include "datadeal.h"

std::vector<int> split(std::string& s, char div)
{
	std::vector<int> tokens;
	std::string token;
	std::istringstream tokenStream(s);

	while (std::getline(tokenStream, token, div))
		tokens.push_back(std::stoi(token));

	return tokens;
}

std::vector<Data> loadData(const std::string filepath)
{
	std::ifstream ifs(filepath, std::ios::in);
	if (!ifs.is_open())
		CatchErr("Failed to open file: " + filepath);
	
	std::vector<Data> dataSet;
	std::string line;
	int row = 0;
	while (std::getline(ifs, line))
	{
		row++;
		if (line.empty())
		{
			std::cerr << "第" << row << "行数据为空，已跳过" << std::endl;
			continue;
		}

		std::vector<int> elements = split(line, ',');
		if (elements.size() < 2)
			CatchErr("第" + std::to_string(row) + "行数据无效");

		Data data;
		data.features.reserve(elements.size() - 1);
		data.features.assign(elements.begin(), elements.end()-1);
		data.tag = *(elements.end() - 1);

		dataSet.push_back(data);
	}

	return dataSet;
}

void printData(std::vector<Data>& dataSet)
{
	for (auto& d : dataSet)
	{
		for (auto& f : d.features)
		{
			std::cout << f << " ";
		}
		std::cout << d.tag << std::endl;
	}
}

std::pair<std::vector<std::vector<int>>, std::vector<int>> transform(const std::vector<Data>& data)
{
	if (data.empty())
		CatchErr("tranform: 数据为空");
	int row_sz = data[0].features.size();
	int col_sz = data.size();
	std::vector<std::vector<int>> features(row_sz, std::vector<int>(col_sz));
	std::vector<int> tags;
	tags.reserve(col_sz);
	for (int j = 0; j < col_sz; ++j)
	{
		for (int i = 0; i < row_sz; ++i)
		{
			features[i][j] = data[j].features[i];
		}
		tags.push_back(data[j].tag);
	}

	return std::make_pair(features, tags);
}

std::vector<int> bincount(const std::vector<int>& buffer)
{
	return unique<true>(buffer).second;
}

std::vector<bool> in(const std::vector<int>& buffer, int element)
{
	std::vector<bool> inlist;
	inlist.reserve(buffer.size());
	for (auto& val : buffer)
		inlist.push_back(val == element);

	return inlist;
}

std::vector<int> splitlist(const std::vector<bool>& idxlist, const std::vector<int>& source)
{
	std::vector<int> dist;
	for (int i = 0; i < idxlist.size(); ++i)
	{
		if (idxlist[i])
			dist.push_back(source[i]);
	}

	return dist;
}

std::vector<std::vector<int>> splitlist(const std::vector<bool>& idxlist, std::vector<std::vector<int>> source)
{
	int offset = 0;
	for (int i = 0; i < idxlist.size(); ++i)
	{
		if (!idxlist[i])
		{
			for (auto& s : source)
			{
				s.erase(s.begin() + i - offset);
			}
			offset++;
		}
	}

	return source;
}