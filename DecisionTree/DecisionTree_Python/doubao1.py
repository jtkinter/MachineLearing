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

def plot_decision_tree(root):
    """绘制决策树，优化文字可读性"""

    # 计算树的深度和每一层的节点数
    def get_tree_info(node, depth=0, layer_counts=None):
        if layer_counts is None:
            layer_counts = {}
        layer_counts[depth] = layer_counts.get(depth, 0) + 1

        max_depth = depth
        for child in node.children:
            child_depth, layer_counts = get_tree_info(child, depth + 1, layer_counts)
            if child_depth > max_depth:
                max_depth = child_depth
        return max_depth, layer_counts

    max_depth, layer_counts = get_tree_info(root)
    total_layers = max_depth + 1

    # 动态调整图的尺寸
    max_nodes_per_layer = max(layer_counts.values()) if layer_counts else 1
    width = 8 + max_nodes_per_layer * 1.5
    height = 6 + total_layers * 2
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95)

    # 存储每个节点的坐标 (x, y)
    node_positions = {}

    def calculate_positions(node, depth=0, parent_pos=None):
        if depth == 0:  # 根节点
            x = 0.5
        else:
            parent_x, _ = parent_pos
            num_children = len(node.parent.children)
            min_interval = 0.05
            max_interval = 0.3
            interval = min(max_interval, max(min_interval, 0.8 / num_children))
            child_index = node.parent.children.index(node)
            start_x = parent_x - (num_children - 1) * interval / 2
            x = start_x + child_index * interval

        y = 0.85 - (depth / (total_layers * 1.1))
        node_positions[node] = (x, y)

        for child in node.children:
            child.parent = node
            calculate_positions(child, depth + 1, (x, y))

    root.parent = None
    calculate_positions(root)

    # 绘制连接线
    def draw_connections(node):
        if node.is_leaf:
            return

        parent_x, parent_y = node_positions[node]
        for child in node.children:
            child_x, child_y = node_positions[child]

            con = ConnectionPatch(
                (parent_x, parent_y - 0.015),
                (child_x, child_y + 0.015),
                coordsA="axes fraction", coordsB="axes fraction",
                arrowstyle="-", color="black", linewidth=1.2
            )
            ax.add_artist(con)

            # 增大分支文字字体，优化背景
            ax.text((parent_x + child_x) / 2, (parent_y + child_y) / 2,
                    f"值: {child.val}",
                    ha="center", va="center", fontsize=10,  # 字体增大到10
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2))  # 背景不透明度提高，padding增加

            draw_connections(child)

    # 绘制节点（增大尺寸，优化文字空间）
    def draw_nodes(node):
        x, y = node_positions[node]

        if node.is_leaf:
            # 增大叶子节点半径，确保文字显示
            node_shape = Circle((x, y), 0.03, facecolor="#90EE90", edgecolor="black", linewidth=1.2)
            ax.add_patch(node_shape)
            ax.text(x, y, f"结果: {node.val}",
                    ha="center", va="center", fontsize=11)  # 字体增大到11
        else:
            # 增大内部节点尺寸，确保文字显示
            node_shape = Rectangle(
                (x - 0.05, y - 0.025),
                0.1, 0.05,
                facecolor="#87CEFA", edgecolor="black", linewidth=1.2
            )
            ax.add_patch(node_shape)
            ax.text(x, y, f"特征: {node.feature}",
                    ha="center", va="center", fontsize=11)  # 字体增大到11

        for child in node.children:
            draw_nodes(child)

    draw_connections(root)
    draw_nodes(root)

    def clean_parent_attr(node):
        if hasattr(node, 'parent'):
            del node.parent
        for child in node.children:
            clean_parent_attr(child)

    clean_parent_attr(root)

    plt.title("决策树可视化", fontsize=16, y=0.98)
    plt.tight_layout()
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
    plot_decision_tree(decision_tree)

    test_features, test_tags = load_data("testset.txt")
    correct = 0
    for i in range(test_features.shape[0]):
        res = predict_type(decision_tree, test_features[i, :])
        if res == test_tags[i]:
            correct += 1

    print(f"模型正确率：{correct/test_features.shape[0]*100:.0f}%")