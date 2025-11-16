import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, ConnectionPatch

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

def get_gain(datas: np.ndarray, tags: np.ndarray):
    sample_num, feature_num = datas.shape
    base_ent = count_ent(np.bincount(tags), sample_num)
    print(f"base_ent: {base_ent}")

    redata = datas.T
    gains = []
    for r in redata:
        tag, cnt = np.unique(r, return_counts=True)
        count_accuracy = 0.0
        for i in range(len(tag)):
            t = tag[i]
            c = cnt[i]
            sub_list = (r == t)
            mask = tags[sub_list]
            count_accuracy += (c/sample_num) * count_ent(np.bincount(mask), c)
        print(f"条件熵：{count_accuracy}，信息增益：{base_ent-count_accuracy}")
        gains.append(base_ent-count_accuracy)

    return np.array(gains)

def create_decision(datas: np.ndarray, tags: np.ndarray, range_list: list[int]):
    if len(np.unique(tags)) == 1: # 结果只有一种情况，无需分类
        return DecisionTreeNode(tags[0], -1, True)
    if len(range_list) == 0: # 特征列表上的数据被用完了，取最大的可能作为叶子节点的分类结果
        max_tag = np.argmax(np.bincount(tags))
        return DecisionTreeNode(max_tag, -1, True)

    # 将所有特征的信息增益算出来
    gain_list = get_gain(datas, tags)
    # 找到最大值下标
    max_idx = np.argmax(gain_list)

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
        children_node = create_decision(sub_datas, sub_tags, new_list)
        children_node.val = v
        children.append(children_node)

    return DecisionTreeNode(max_idx, range_list[max_idx], is_leaf=False, children=children)

# test
def get_hierarchy_tree(root: DecisionTreeNode):
    if not root:
        return []
    plot_list = []
    queue_list = [root]

    while queue_list:
        level_list = []
        sz = len(queue_list)
        for _ in range(sz):
            node = queue_list.pop(0)
            if type(node) == list:
                level_list.append([])
                queue_list.append([])
            elif node.is_leaf:
                level_list.append(str(node))
                queue_list.append([])
            else:
                level_list.append(str(node))
                queue_list.extend(node.children)
        plot_list.append(level_list)
        tag = 0
        for q in queue_list:
            if type(q) == list:
                tag += 1
        if tag == len(queue_list):
            break
    return plot_list

def plot_decision_tree(plot_list):
    level = len(plot_list)
    height = 8 + len(plot_list[-1])*1
    width = 6 + level*0.5
    fig, ax = plt.subplots(figsize=(height, width))

    node_positions = []
    for col_idx in range(level):
        level_node = []
        y = (level-col_idx)/level-0.05
        sz = len(plot_list[col_idx])
        low = 0.5-(sz*0.5)/width
        for row_idx in range(sz):
            if type(plot_list[col_idx][row_idx]) == list:
                level_node.append([])
                continue
            x = low+row_idx/width
            level_node.append([plot_list[col_idx][row_idx], x, y])

            node_shape = Rectangle(
                (x - 0.05, y - 0.025),
                0.1, 0.05,
                facecolor="#87CEFA", edgecolor="black", linewidth=1.2
            )
            ax.add_patch(node_shape)
            ax.text(x, y, plot_list[col_idx][row_idx],
                    ha="center", va="center", fontsize=11)  # 字体增大到11
        node_positions.append(level_node)
    print(node_positions)

    ax.axis("off")
    plt.title("决策树可视化", fontsize=16)
    plt.show()

def predict_type(root: DecisionTreeNode, test):
    if root.is_leaf:
        return root.val

    test_val = test[root.feature]
    for child in root.children:
        if child.val == test_val:
            return predict_type(child, test)

    return None

if __name__ == "__main__":

    features, tags = load_data("dataset.txt")
    decision_tree = create_decision(features, tags, [x for x in range(features.shape[1])])
    plot_data = get_hierarchy_tree(decision_tree)
    plot_decision_tree(plot_data)

    test_features, test_tags = load_data("testset.txt")
    correct = 0
    for i in range(test_features.shape[0]):
        res = predict_type(decision_tree, test_features[i, :])
        if res == test_tags[i]:
            correct += 1

    print(f"模型正确率：{correct/test_features.shape[0]*100:.0f}%")