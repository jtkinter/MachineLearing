import matplotlib.pyplot as plt
from src.utils.common_import import np
from src.data.dataLoader import load_data
from src.model.decision_tree import create_decision_tree_pre_pruning, create_decision_tree_post_pruning
from src.model.calculate import gain, gain_ratio, cart
from src.show.plotshow import plot_decision_tree, tree_height, plot_heatmap
from src.evaluation.evaluator import get_heatmap, evaluate

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

if __name__ == "__main__":
    train_features, train_tags = load_data("dataSource/dataset.txt")

    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(train_features))
    train_features = train_features[shuffle_indices]
    train_tags = train_tags[shuffle_indices]
    # 留出
    split_num = int(len(train_features)*2/5)
    verify_features = train_features[:split_num]
    verify_tags = train_tags[:split_num]

    train_features = train_features[split_num:]
    train_tags = train_tags[split_num:]

    tag_list = list(range(len(train_tags)))
    pre_id3_tree = create_decision_tree_pre_pruning(train_features, train_tags, tag_list, verify_features, verify_tags, gain)
    pre_c4_5_tree = create_decision_tree_pre_pruning(train_features, train_tags, tag_list, verify_features, verify_tags, gain_ratio)
    pre_cart_tree = create_decision_tree_pre_pruning(train_features, train_tags, tag_list, verify_features, verify_tags, cart)

    back_id3_tree = create_decision_tree_post_pruning(train_features, train_tags, tag_list, verify_features, verify_tags, gain)
    back_c4_5_tree = create_decision_tree_post_pruning(train_features, train_tags, tag_list, verify_features, verify_tags, gain_ratio)
    back_cart_tree = create_decision_tree_post_pruning(train_features, train_tags, tag_list, verify_features, verify_tags, cart)

    models = {
        "pre_ID3": pre_id3_tree,
        "pre_C4.5": pre_c4_5_tree,
        "pre_CART": pre_cart_tree,
        "post_ID3": back_id3_tree,
        "post_C4.5": back_c4_5_tree,
        "post_CART": back_cart_tree
    }

    fig_height = max(tree_height(x) for _, x in models.items())
    tree_width = 8 + pow(2, fig_height - 1) * 1.5
    fig, axs =plt.subplots(6,2, figsize=(tree_width + 6, 30))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.suptitle("决策树算法对比", fontsize=20, y=0.95)

    row_name_y = 0.9
    test_features, test_tags = load_data("dataSource/testset.txt")
    for i, (name, tree) in enumerate(models.items(), 0):
        fig.text(0.02, row_name_y-i*0.16, name, fontsize=16, ha='center', va='center', rotation='vertical')
        plot_decision_tree(tree, tree_height(tree), axs[i, 0])
        heatmap, tag_text = get_heatmap(tree, train_tags, test_features, test_tags)
        plot_heatmap(heatmap, tag_text, axs[i, 1])
        evaluate(name, heatmap, tag_text)
    plt.tight_layout()
    plt.show()