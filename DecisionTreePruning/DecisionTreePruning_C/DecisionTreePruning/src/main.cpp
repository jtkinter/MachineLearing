#include "datadeal.h"
#include "decisionTree_algorithm.h"
#include "Evaluate.h"
#include "plotshow.h"

struct Model
{
    std::string name;
    ClassifierFunc cls;
    Strategy strategy;
};

int main()
{
    namespace plt = matplot;

    auto start = std::chrono::steady_clock::now();
    auto data = loadData("source/dataset.txt");
    //data_shuffle(data);
    int sz = data.size()*2/5;
    std::vector<Data> train_data(data.begin() + sz, data.end());
    std::vector<Data> verify_data(data.begin(), data.begin() + sz);

    auto [train_features, train_tags] = transform(train_data);
    auto [verify_features, verify_tags] = transform(verify_data);

    std::vector<int> range;
    range.reserve(train_tags.size());
    for (int i = 0; i < train_tags.size(); ++i)
        range.push_back(i);

    std::vector <Model> map =
    {
        {"pre_ID3", Gain, PrePruning},
        {"post_ID3", Gain, PostPruning},
        {"pre_C4.5", GainRatio, PrePruning},
        {"post_C4.5", GainRatio, PostPruning},
        {"pre_CART", CART, PrePruning},
        {"post_CART", CART, PostPruning}
    };
    
    plt::figure(true)->size(3000, 1800);
    auto tests = loadData("source/testset.txt");
    auto [_, tags_t] = transform(tests);

    for (int i = 0; i < map.size(); ++i)
    {
        auto ax1 = plt::subplot(3, 4, i*2);
        DecisionTreeNode* tree = map[i].strategy(train_features, train_tags, range, map[i].cls, verify_features, verify_tags);
        plotDecisionTree(tree, ax1);

        // test
        auto [cfmt, reverse_tags] = ConfusionMatrix(tree, tests, tags_t);
        auto evaluation = Evaluate(cfmt, reverse_tags);
        std::cout << "使用" << map[i].name << "分类的模型准确率为" << Accurancy(cfmt) * 100 << "%" << std::endl;
        for (const auto& e : evaluation)
        {
            std::cout << "类别" << e.first << "-> 精确率: " << e.second.first*100 << "%，召回率: " << e.second.second*100 << "%" << std::endl;
        }
        auto ax2 = plt::subplot(3, 4, i*2+1);
        plotHeatmap(cfmt, reverse_tags, ax2);

        delete tree;
    }

    plt::show();

	return 0;
}