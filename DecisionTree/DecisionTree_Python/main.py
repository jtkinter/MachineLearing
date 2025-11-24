import matplotlib.pyplot as plt
from src.data.dataLoader import load_data
from src.model.decision_tree import create_decision_tree
from src.model.calculate import gain, gain_ratio, cart
from src.show.plotshow import plot_decision_tree, tree_height, plot_heatmap
from src.evaluation.evaluator import get_heatmap, evaluate

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

if __name__ == "__main__":
    train_features, train_tags = load_data("dataSource/dataset.txt")
    tag_list = list(range(len(train_tags)))
    id3_tree = create_decision_tree(train_features, train_tags, tag_list, gain)
    c4_5_tree = create_decision_tree(train_features, train_tags, tag_list, gain_ratio)
    cart_tree = create_decision_tree(train_features, train_tags, tag_list, cart)

    models = {
        "ID3": id3_tree,
        "C4.5": c4_5_tree,
        "CART": cart_tree
    }

    fig_height = max(tree_height(x) for _, x in models.items())
    tree_width = 8 + pow(2, fig_height - 1) * 1.5
    fig, axs =plt.subplots(3,2, figsize=(tree_width + 6, 18))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle("决策树算法对比", fontsize=20, y=0.95)

    row_name_y = 0.8
    test_features, test_tags = load_data("dataSource/testset.txt")
    for i, (name, tree) in enumerate(models.items()):
        fig.text(0.02, row_name_y-i*0.3, name, fontsize=16, ha='center', va='center', rotation='vertical')
        plot_decision_tree(tree, tree_height(tree), axs[i, 0])
        heatmap, tag_text = get_heatmap(tree, train_tags, test_features, test_tags)
        plot_heatmap(heatmap, tag_text, axs[i, 1])
        evaluate(name, heatmap, tag_text)
    plt.tight_layout()
    plt.show()