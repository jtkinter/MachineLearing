#include "datadeal.h"

inline vector<string> getVec(const string& line, const char c)
{
	stringstream ss(line);
	string token;
	vector<string> portion;
	while (getline(ss, token, c))
		portion.push_back(token);
	if (line.back() == c)
		portion.push_back("");
	return portion;
}

vector<Data> loadData(const string& filepath)
{
	ifstream ifs(filepath, ios::in);
	Assert(ifs.is_open(), "loadData: failed to open the file -> " + filepath);
	
	vector<Data> dataSet;
	string line;
	while (getline(ifs, line))
	{
		Data data;
		vector<string> portion = getVec(line, ',');
		for (int i = 0; i < portion.size()-1; ++i)
		{
			double e = stof(portion[i]);
			if (e == (int)e)
				data.disperse.push_back(e);
			else
				data.continuous.push_back(e);
		}
		if (portion.back().empty())
			data.tag = -1;
		else
			data.tag = stoi(portion.back());

		dataSet.push_back(data);
	}
	
	return dataSet;
}

pair<vector<vector<int>>, vector<vector<double>>> transform(const vector<Data>& data)
{
	Assert(!data.empty(), "transform: data is empty");

	int num = data.size();
	int sz = data[0].disperse.size();
	vector<vector<int>> disperse(sz + 1, vector<int>(num));
	vector<vector<double>> continuous(data[0].continuous.size(), vector<double>(num));

	for (int i = 0; i < num; ++i)
	{
		for (int j = 0; j < sz; ++j)
			disperse[j][i] = data[i].disperse[j];
		for (int j = 0; j < data[0].continuous.size(); ++j)
			continuous[j][i] = data[i].continuous[j];
		disperse[sz][i] = data[i].tag;
	}

	return make_pair(disperse, continuous);
}
