#include "datadeal.h"
#include "decisionTree_algorithm.h"
#include "Evaluate.h"
#include "plotshow.h"

int main()
{
    namespace plt = matplot;

    auto data = loadData("source/dataset.txt");
    auto [features, tags] = transform(data);
    std::vector<int> range;
    range.reserve(features.size());
    for (int i = 0; i < features.size(); ++i)
        range.push_back(i);

    std::vector<std::pair<std::string, ClassifierFunc>> map = 
    {
        std::make_pair("ID3", Gain),
        std::make_pair("C4.5", GainRatio),
        std::make_pair("GINI", CART)
    };
    
    plt::figure(true)->size(1800, 1800);
    auto tests = loadData("source/testset.txt");
    auto [_, tags_t] = transform(tests);

    for (int i = 0; i < map.size(); ++i)
    {
        auto ax1 = plt::subplot(3, 2, i*2);
        auto tree = CreateDecistionTree(features, tags, range, map[i].second);
        plotDecisionTree(tree, ax1);

        // test
        auto [cfmt, reverse_tags] = ConfusionMatrix(tree, tests, tags_t);
        auto evaluation = Evaluate(cfmt, reverse_tags);
        std::cout << "使用"+map[i].first + "分类的模型准确率为" << Accurancy(cfmt) * 100 << "%" << std::endl;
        for (const auto& e : evaluation)
        {
            std::cout << "类别" << e.first << "-> 精确率: " << e.second.first*100 << "%，召回率: " << e.second.second*100 << "%" << std::endl;
        }
        auto ax2 = plt::subplot(3, 2, i * 2 + 1);
        plotHeatmap(cfmt, reverse_tags, ax2);

        delete tree;
    }
    plt::show();

	return 0;
}