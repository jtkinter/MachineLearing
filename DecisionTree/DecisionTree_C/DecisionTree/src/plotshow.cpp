#include "plotshow.h"

#include "decisionTree_algorithm.h"

void plotDecisionTree(DecisionTreeNode* root, matplot::axes_handle ax)
{
	using namespace matplot;

	std::queue<DecisionTreeNode*> tree_queue;
	tree_queue.push(root);
	std::unordered_map<DecisionTreeNode*, std::pair<double, double>> nodepositions;
	int level = 0;
	int max_level = getLevel(root);

	ax->hold(on);
	while (!tree_queue.empty())
	{
		int sz = tree_queue.size();
		level++;
		double branchGap = 0.5 - 0.13 * level;
		double layer = 1.0 / max_level;
		while (sz--)
		{
			DecisionTreeNode* node = tree_queue.front();
			tree_queue.pop();
			if (level == 1)
			{
				double x = 0.5;
				double y = (max_level - level + 1) / static_cast<double>(max_level)-0.05;
				nodepositions.insert({ node, std::make_pair(x, y) });
			}
			if (!node->is_leaf)
			{
				double parent_x = nodepositions[node].first;
				double parent_y = nodepositions[node].second;
				double low = parent_x - branchGap * (node->Children.size() - 1) / 2;
				
				for (auto& child : node->Children)
				{
					double y = parent_y - layer;
					nodepositions.insert({ child, std::make_pair(low, y) });
					ax->plot({ parent_x, low }, { parent_y, y })->color("black").line_width(1.2);
					ax->text((parent_x + low) / 2, (parent_y + y) / 2, std::to_string(child->val))->color("r").font_size(11);
					low += branchGap;
					tree_queue.push(child);
				}
			}
		}
	}

	for (auto& node : nodepositions)
	{
		double x = node.second.first;
		double y = node.second.second;
		if (node.first->is_leaf)
		{
			double radius = 0.035;
			std::vector<double> theta;
			for (double t = 0; t <= 2 * pi; t += pi / 50)
				theta.push_back(t);
			std::vector<double> cx, cy;
			for (double t : theta)
			{
				cx.push_back(x + radius * cos(t));
				cy.push_back(y + radius * sin(t));
			}
			ax->plot(cx, cy, "-r")->color("g").line_width(1.5).fill("g");
		}
		else
		{
			ax->rectangle(x - 0.025, y - 0.05, 0.05, 0.1)->color("c").fill("c");
		}
		ax->text(x - 0.02, y - 0.005, node.first->to_string())->font_size(11);
	}
	ax->hold(off);

	for (const auto& entry : nodepositions)
	{
		std::cout << *entry.first << "(x: " << entry.second.first << " y: " << entry.second.second << ")" << std::endl;
	}

	ax->axis(false);
}

void plotHeatmap(std::vector<std::vector<int>>& confunsion, std::unordered_map<int, int> reverse_map, matplot::axes_handle ax)
{
	using namespace matplot;
	ax->heatmap(confunsion)->normalization(matrix::color_normalization::columns);
	int sz = reverse_map.size();
	std::vector<std::string> xticks, yticks;
	xticks.reserve(sz), yticks.reserve(sz);
	for (int i = 0; i < sz; ++i)
	{
		xticks.push_back(std::to_string(reverse_map[i]));
		yticks.push_back(std::to_string(reverse_map[i]));
	}
	ax->x_axis().ticklabels(xticks);
	ax->y_axis().ticklabels(yticks);
}