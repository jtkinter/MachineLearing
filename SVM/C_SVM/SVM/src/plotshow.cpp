#include "plotshow.h"

#include "datadeal.h"
using namespace matplot;

// 一组点的分布
void dotshow(const std::vector<double>& dot_x, const std::vector<double>& dot_y, std::string color = "blue", int scale = 4,
	enum line_spec::marker_style marker = line_spec::marker_style::point)
{
	auto l = scatter(dot_x, dot_y, scale);
	l->marker_style(marker);
	l->marker_color(color);
}

void distrshow(const std::unordered_map<double, std::pair<std::vector<double>, std::vector<double>>>& samples)
{
	std::unordered_map<double, std::string> colors =
	{
		{-1, "red"},
		{ 1, "blue"}
	};

	hold(on);
	for (const auto& port : samples)
		dotshow(port.second.first, port.second.second, colors[port.first]);
	hold(off);
	show();
}

void supportshow(const std::unordered_map<double, std::pair<std::vector<double>, std::vector<double>>>& samples)
{
	std::unordered_map<double, std::string> colors =
	{
		{-1, "red"},
		{ 1, "blue"}
	};

	hold(on);
	for (const auto& port : samples)
		dotshow(port.second.first, port.second.second, colors[port.first], 8, line_spec::marker_style::asterisk);
	hold(off);
	show();
}