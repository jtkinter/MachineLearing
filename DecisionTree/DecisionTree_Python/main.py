import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
from numpy.f2py.auxfuncs import throw_error
from numpy.ma.core import zeros

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

class DecisionTreeNode:
    def __init__(self, val, feature, is_leaf=False, children=None):
        self.val = val
        self.feature = feature
        self.is_leaf = is_leaf
        self.children = children if children is not None else []

    def __str__(self):
        if self.is_leaf:
            return f"Result: {self.val}"
        return f"Node: {self.feature}"

def load_data(filepath):
    data = []
    tag = []
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 5:
                print(f"特征数据无法转换：{line}")
                continue
            data.append([int(x) for x in parts[:-1]])
            tag.append(int(parts[-1]))

    data_np = np.array(data, dtype=np.int32)
    tag_np = np.array(tag, dtype=np.int32)
    return data_np, tag_np

def count_ent(cnt, num):
    prob = np.array(cnt, dtype=float)
    prob /= num
    prob = prob[prob>0]
    return -np.sum(prob*np.log2(prob))

def get_gain(datas: np.ndarray, tags: np.ndarray, ratio = False):
    sample_num, feature_num = datas.shape
    base_ent = count_ent(np.bincount(tags), sample_num)
    print(feature_num)
    # print(f"base_ent: {base_ent}")

    gains = []
    feature_ent = []
    for r in datas.T:
        tag, cnt = np.unique(r, return_counts=True)
        count_accuracy = 0.0
        for t, c in zip(tag, cnt):
            mask = tags[r == t]
            count_accuracy += (c/sample_num) * count_ent(np.bincount(mask), c)
        print(f"条件熵：{count_accuracy}，信息增益：{base_ent-count_accuracy}")
        if ratio:
            feature_ent.append(count_ent(cnt, sample_num))
        gains.append(base_ent - count_accuracy)
    gains = np.array(gains)
    if ratio:
        gain_ratios = gains/np.array(feature_ent)
        return gain_ratios
    else:
        return gains

def count_gini(cnt, num):
    prob = np.array(cnt, dtype=float) / num
    return 1 - np.sum(prob**2)

def get_gini(datas: np.ndarray, tags: np.ndarray):
    sample_num, feature_num = datas.shape
    gini = []
    for r in datas.T:
        tag, cnt = np.unique(r, return_counts=True)
        cnt_gini = 0.0
        for t, c in zip(tag, cnt):
            mask = tags[r == t]
            cnt_gini += (c/sample_num) * count_gini(np.bincount(mask), c)
        gini.append(cnt_gini)
        print(f"Gini：{cnt_gini}")


    return np.array(gini)

def create_ent_decision(datas: np.ndarray, tags: np.ndarray, range_list: list[int], classifier=get_gain, ratio = False):
    if len(np.unique(tags)) == 1: # 结果只有一种情况，无需分类
        return DecisionTreeNode(tags[0], -1, True)
    if len(range_list) == 0: # 特征列表上的数据被用完了，取最大的可能作为叶子节点的分类结果
        max_tag = np.argmax(np.bincount(tags))
        return DecisionTreeNode(max_tag, -1, True)

    max_idx = -1
    if classifier != get_gini:
        # 将所有特征的信息增益算出来
        gain_list = classifier(datas, tags, ratio)
        # 找到最大值下标
        max_idx = np.argmax(gain_list)
    else:
        gain_list = classifier(datas, tags)
        max_idx = np.argmin(gain_list)

    # 获取最大值列的所有数据
    feature_value = datas[:, max_idx]
    # 统计不同种类的个数
    queue_list = np.unique(feature_value)

    # 删除原数据的对应列
    new_datas = np.delete(datas, obj=max_idx, axis=1)
    # 深拷贝特征列表
    new_list = copy.deepcopy(range_list)
    # 删除特征列表中选中的特征
    new_list.pop(max_idx)

    # 以当前节点为根，生成出当前分类下的子节点
    children = []
    for v in queue_list:
        # 获取当前种类的下标列表
        sub_list = (feature_value == v)
        # 获取当前分类下数据
        sub_datas = new_datas[sub_list]
        sub_tags = tags[sub_list]
        # 生成子节点
        children_node = create_ent_decision(sub_datas, sub_tags, new_list, classifier)
        children_node.val = v
        children.append(children_node)

    return DecisionTreeNode(max_idx, range_list[max_idx], is_leaf=False, children=children)

def get_max_level(root):
    queue_list = [root]
    max_level = 0
    while queue_list:
        sz = len(queue_list)
        max_level += 1
        for _ in range(sz):
            node = queue_list.pop(0)
            if not node.is_leaf:
                queue_list.extend(node.children)
    return max_level

def plot_decision_tree(root, max_level, ax):

    # 计算点的位置，同时画线
    queue_list = [root]
    node_positions = {}
    level_long = 1/max_level
    level = 0
    while queue_list:
        sz = len(queue_list)
        level += 1
        branch_gap = 0.5-0.13*level
        for _ in range(sz):
            node = queue_list.pop(0)
            if level == 1:
                x = 0.5
                y = (max_level - level + 1) / max_level - 0.1
                node_positions[node] = (x, y)
            if not node.is_leaf:
                parent_x, parent_y = node_positions[node]
                low = parent_x - branch_gap*(len(node.children)-1)/2
                for child in node.children:
                    x = low
                    low += branch_gap
                    y = parent_y - level_long
                    node_positions[child] = (x, y)

                    con = ConnectionPatch(
                        (parent_x, parent_y - 0.015),
                        (x, y + 0.015),
                        coordsA="axes fraction", coordsB="axes fraction",
                        arrowstyle="-", color="black", linewidth=1.2
                    )
                    ax.add_artist(con)
                    ax.text((parent_x + x) / 2, (parent_y + y) / 2,
                            f"值: {child.val}",
                            ha="center", va="center", fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2))

                queue_list.extend(node.children)

    # 画点
    for node, (x, y) in node_positions.items():
        if node.is_leaf:
            node_shape = Circle((x, y), 0.035, facecolor="#90EE90", edgecolor="black", linewidth=1.2)
            ax.add_patch(node_shape)
            ax.text(x, y, f"结果: {node.val}", ha="center", va="center", fontsize=11)
        else:
            node_shape = Rectangle(
                (x - 0.05, y - 0.025),
                0.1, 0.05,
                facecolor="#87CEFA", edgecolor="black", linewidth=1.2
            )
            ax.add_patch(node_shape)
            ax.text(x, y, f"特征: {node.feature}", ha="center", va="center", fontsize=9)

    ax.axis("off")

def predict_type(root: DecisionTreeNode, test):
    if root.is_leaf:
        return root.val

    test_val = test[root.feature]
    for child in root.children:
        if child.val == test_val:
            return predict_type(child, test)

    return None

# 获取准确率
def get_correct(root, test_f, test_t):
    correct = 0
    for i in range(test_f.shape[0]):
        res = predict_type(root, test_f[i, :])
        if res == test_t[i]:
            correct += 1
    return correct

# 获取精确率和召回率
def get_predict(root, test_f, test_t, train_tags):
    true_list = test_t.tolist()
    majority = np.argmax(np.bincount(train_tags))
    predict = []

    for test in test_f:
        res = predict_type(root, test)
        if res is None:
            res = majority
        predict.append(res)

    classes = np.unique(true_list)
    precision_dict = {}
    recall_dict = {}
    for cls in classes:
        cls = cls.item()
        tp, fp, fn = 0, 0, 0
        for p, t in zip(predict, true_list):
            tp += 1 if (p == cls) and (t == cls) else 0
            fp += 1 if (p == cls) and (t != cls) else 0
            fn += 1 if (p != cls) and (t == cls) else 0
        precision_dict[cls] = tp/(tp+fp) if tp+fp != 0.0 else 0.0
        recall_dict[cls] = tp/(tp+fn) if tp+fn != 0.0 else 0.0
    # 这里-1定义为平均精确率
    precision_dict[-1] = np.mean(list(precision_dict.values())).item()
    recall_dict[-1] = np.mean(list(recall_dict.values())).item()
    return precision_dict, recall_dict

def get_confusion(root, test_f, test_t):
    unique_tags = np.unique(test_t)
    tag_map = {tag:i for i, tag in enumerate(unique_tags)} # 混淆矩阵的范围是0-1，标签范围是0-3，需要映射
    sz = len(unique_tags)
    mat_cf = zeros((sz, sz))
    for i in range(test_f.shape[0]):
        res = predict_type(root, test_f[i, :])
        if res in unique_tags:
            mat_cf[tag_map[test_t[i]]][tag_map[res]] += 1
    return mat_cf, unique_tags

def print_dict(dictionary, name):
    for key, value in dictionary.items():
        if key == -1:
            print(f"平均{name}为：{value}")
        else:
            print(f"类别{key}{name}的为：{value}")

# 绘制热力图
def plot_heatmap(heat_conf, unique_tags, ax):
    class_names = list(unique_tags)
    im = ax.imshow(heat_conf)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(labels=class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(labels=class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = heat_conf[i, j]
            text_color = "black" if im.norm(value) > 0.5 else "white"
            ax.text(j, i, value, ha="center", va="center", color=text_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sample Count")

if __name__ == "__main__":
    # 生成决策树
    features, tags = load_data("dataset.txt")
    # features = np.delete(features, 0, axis=1)

    # 使用信息增益的方式生成 - ID3
    decision_tree1 = create_ent_decision(features, tags, list(range(features.shape[1])))
    # 使用信息增益比生成 - C4.5
    decision_tree2 = create_ent_decision(features, tags, list(range(features.shape[1])), get_gain, True)
    # 使用基尼系数生成 - Gini
    decision_tree3 = create_ent_decision(features, tags, list(range(features.shape[1])), get_gini)

    test_features, test_tags = load_data("testset.txt")
    # test_features = np.delete(test_features, 0, axis=1)

    models = [
        (decision_tree1, "ID3"),
        (decision_tree2, "C4.5"),
        (decision_tree3, "Gini")
    ]
    # 调整画布大小
    max_level = max(get_max_level(x) for x, _ in models)
    tree_width = 8 + pow(2, max_level - 1) * 1.5
    fig, axs = plt.subplots(3, 2, figsize=(tree_width + 6, 18))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    # 整体标题
    fig.suptitle("决策树算法对比", fontsize=20, y=0.95)
    # 左侧垂直标注算法名称
    fig.text(0.02, 0.8, "ID3", fontsize=16, ha='center', va='center', rotation='vertical')
    fig.text(0.02, 0.5, "C4.5", fontsize=16, ha='center', va='center', rotation='vertical')
    fig.text(0.02, 0.2, "Gini", fontsize=16, ha='center', va='center', rotation='vertical')

    for i, (dtree, func) in enumerate(models):
        print(f"这是来自{func}生成的决策树的测试")
        # 测试模型
        accuracy = get_correct(dtree, test_features, test_tags)/test_features.shape[0]*100
        precision, recallment = get_predict(dtree, test_features, test_tags, tags)
        cf, queue = get_confusion(dtree, test_features, test_tags)

        # 结果可视化
        print(f"模型准确率：{accuracy:.0f}%")
        print_dict(precision, "精确率")
        print_dict(recallment, "召回率")
        print()

        # 决策树可视化
        plot_decision_tree(dtree, max_level, axs[i, 0])
        # 热力图可视化
        plot_heatmap(cf, queue, axs[i, 1])

    plt.tight_layout()
    plt.show()