#include "datdeal.h"

#include <random>

std::vector<Data> loadData(const std::string& filepath)
{
	std::ifstream ifs;
	ifs.open(filepath, std::ios::in);
	if (!ifs.is_open())
		throw std::invalid_argument("Failed to open file: " + filepath);

	std::vector<Data> datalist;
	double trail, game, eating;
	std::string tag;

	std::vector<std::string> searchlist =
	{
		"largeDoses",
		"smallDoses",
		"didntLike"
	};

	while (ifs >> trail >> game >> eating >> tag)
	{
		int type = -1;
		for (int i = 0; i < searchlist.size(); ++i)
		{
			if (searchlist[i] == tag)
			{
				type = i;
				break;
			}
		}
		datalist.push_back({ trail, game, eating, type });
	}
	ifs.close();

	return datalist;
}

// 更新归一化数据
void calculate(std::vector<double>& feature)
{
	double max = *max_element(feature.begin(), feature.end());
	double min = *min_element(feature.begin(), feature.end());

	double range = max - min;
	for (double& val : feature)
		val = (val - min) / range;
}

// 归一化
void normalized(std::vector<Data>& data)
{
	if (data.empty())
	{
		std::cout << "nullptr doesn't normalized!" << std::endl;
		__debugbreak();
		return;
	}

	std::vector<std::pair<
		std::function<double(const Data&)>,
		std::function<void(Data&, double)>
		>> field =
	{
		{
			[](const Data& d) {return d.game_time; },
			[](Data& d, double val) {d.game_time = val; }
		},
		{
			[](const Data& d) {return d.ice_crime_eating; },
			[](Data& d, double val) {d.ice_crime_eating = val; }
		},
		{
			[](const Data& d) {return d.length; },
			[](Data& d, double val) {d.length = val; }
		}
	};

	for (auto& f : field)
	{
		auto& getVal = f.first;
		auto& setVal = f.second;

		std::vector<double> values;
		for (const auto& d : data)
			values.push_back(getVal(d));

		calculate(values);
		for (int i = 0; i < data.size(); ++i)
			setVal(data[i], values[i]);
	}
}

// 保存数据
void saveData(std::vector<Data>& data, const std::string& savepath)
{
	if (data.empty())
	{
		std::cout << "data is nullptr!" << std::endl;
		__debugbreak();
		return;
	}

	std::ofstream ofs;
	ofs.open(savepath, std::ios::out);
	if (!ofs.is_open())
	{
		std::cout << "save file path doesn't open!" << std::endl;
		return;
	}

	std::vector<std::string> searchlist =
	{
		"largeDoses",
		"smallDoses",
		"didntLike"
	};

	for (const auto& d : data)
	{
		ofs << d.length << " "
			<< d.game_time << " "
			<< d.ice_crime_eating << " "
			<< (d.type == -1 ? "unknown" : searchlist[d.type]) << std::endl;
	}
	ofs.close();

	std::cout << "save successful!" << std::endl;
}

// 打乱数据
std::vector<Data> shuffleData(std::vector<Data>& data)
{
	auto shuffle_data = std::vector<Data>(data.begin(), data.end());
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(shuffle_data.begin(), shuffle_data.end(), g);

	return shuffle_data;
}