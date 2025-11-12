#include "matplot_show.h"
#include <vector>

void plotROC(std::vector<std::pair<std::vector<double>, std::vector<double>>> data)
{
	using namespace matplot;
	auto [fprs1, tprs1] = data[0];
	auto [fprs2, tprs2] = data[1];
	auto [fprs3, tprs3] = data[2];
	//plt::stairs(fprs, tprs);
	auto p = plot(fprs1, tprs1, fprs2, tprs2, fprs3, tprs3, std::vector{0,1}, std::vector{0,1});
    p[0]->line_width(2).line_style("-").color("blue");
    p[1]->line_width(2).line_style("-").color("yellow");
    p[2]->line_width(2).line_style("-").color("green");
    p[3]->line_width(1).line_style("--").color("red");
	xlabel("False Positive Rate");
	ylabel("True Positive Rate");
	title("ROC");
	xlim({ 0, 1 });
	ylim({ 0, 1 });
	show();
}

//void plotROC(std::vector<std::pair<std::vector<double>, std::vector<double>>> data, std::vector<double> aucs, std::vector<std::vector<int>>& list)
//{
//	using namespace matplot;
//	std::vector<std::string> colorList({ "blue", "yellow", "green", "red" });
//	std::vector<std::string> tagName({ "largeDoses", "smallDoses", "didntLike", "Random Guess" });
//
//	subplot(1, 2, 1);
//	hold(on);
//	for (int i = 0; i < data.size(); ++i)
//	{
//		plot(data[i].first, data[i].second)->line_width(2).line_style("-").color(colorList[i]);
//	}
//	plot(std::vector{ 0,1 }, std::vector{ 0,1 })->line_width(1).line_style("--").color(colorList[3]);
//	hold(off);
//
//	xlabel("False Positive Rate");
//	ylabel("True Positive Rate");
//	title("ROC");
//	xlim({ 0, 1 });
//	ylim({ 0, 1 });
//	for(int i = 0; i < aucs.size(); ++i)
//		text(0.02, 0.02+i/20.0, tagName[i] + " AUC: " + std::to_string(aucs[i]));
//	auto l = ::matplot::legend(tagName);
//	l->location(legend::general_alignment::bottomright);
//	l->num_rows(2);
//
//	subplot(1, 2, 2);
//	heatmap(list)->normalization(matrix::color_normalization::columns);
//	title("Three distribution");
//	auto ax = gca();
//	ax->x_axis().ticklabels({ "largeDoses", "smallDoses" , "didntLike" });
//	ax->y_axis().ticklabels({ "largeDoses", "smallDoses" , "didntLike" });
//
//	save("ROC.png");
//	show();
//}

void plotEvalution(std::vector<std::pair<std::vector<double>, std::vector<double>>> data, std::vector<double> aucs, std::vector<std::vector<int>>& list)
{
    using namespace matplot;
    std::vector<std::string> colorList({ "blue", "yellow", "green", "red" });
    std::vector<std::string> tagName({ "largeDoses", "smallDoses", "didntLike", "Random Guess" });

    auto fig = figure(true); // 设置为true是能正常使用figure的关键
    fig->size(2560, 1000);

    subplot(1, 2, 1);
    hold(on);
    for (int i = 0; i < data.size(); ++i)
    {
        plot(data[i].first, data[i].second)->line_width(2).line_style("-").color(colorList[i]);
    }
    plot(std::vector{ 0,1 }, std::vector{ 0,1 })->line_width(1).line_style("--").color(colorList[3]);
    hold(off);

    xlabel("False Positive Rate");
    ylabel("True Positive Rate");
    title("ROC");
    xlim({ 0, 1 });
    ylim({ 0, 1 });

    for (int i = 0; i < aucs.size(); ++i)
        text(0.02, 0.1 - i * 0.025, tagName[i] + " AUC: " + std::to_string(aucs[i]))->font_size(8);
    auto l = ::matplot::legend(tagName);
    l->location(legend::general_alignment::bottomright);
    l->num_rows(2);
    l->font_size(5);

    subplot(1, 2, 2);
    heatmap(list)->normalization(matrix::color_normalization::columns);
    title("Three distribution");
    auto ax = gca();
    ax->x_axis().ticklabels({ "largeDoses", "smallDoses" , "didntLike" });
    ax->y_axis().ticklabels({ "largeDoses", "smallDoses" , "didntLike" });
    ax->x_axis().label_font_size(5);
    ax->y_axis().label_font_size(5);

    //save("ROC.png");
    show();
}


//void plotList(std::vector<std::vector<int>>& data)
//{
//	using namespace matplot;
//	heatmap(data)->normalization(matrix::color_normalization::columns);
//	title("Three distribution");
//	auto ax = gca();
//	ax->x_axis().ticklabels({ "largeDoses", "smallDoses" , "didntLike" });
//	ax->y_axis().ticklabels({ "largeDoses", "smallDoses" , "didntLike" });
//
//	show();
//}