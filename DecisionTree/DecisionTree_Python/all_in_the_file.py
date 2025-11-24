import copy
import numpy as np
from numpy.ma.core import zeros
from collections import deque
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

class DecisionTreeNode:
    def __init__(self, feature, val, children = None):
        self.feature = feature # 非叶子节点存放特征
        self.val = val # 非叶子节点存放feature的类别，叶子节点存放预测值
        self.children = children
        self.is_leaf = children is None

# 导入数据
def load_data(filepath: str):
    features = []
    tags = []
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [int(x) for x in line.split(',')]
            features.append(parts[:-1])
            tags.append(parts[-1])
    np_features = np.array(features)
    np_tags = np.array(tags)
    return np_features, np_tags

# 信息熵算法
def ent(cnt: np.ndarray, num: int) -> float:
    prob = cnt.astype(float)/num
    prob = prob[prob>0]
    return -np.sum(prob*np.log2(prob))

# 总体基尼系数 基尼不纯度
def gini(cnt: np.ndarray, num: int) -> float:
    prob = cnt.astype(float)/num
    prob = prob[prob>0]
    return 1-np.sum(prob*prob)

# 评价数据列表
def val_list(features: np.ndarray, tags: np.ndarray,
             func: callable([[np.ndarray, int], float]), iv = False
             ) -> list[float] | tuple[list[float]]:
    if features is None:
        return []
    sample_num, feature_num = features.shape
    base_ent = func(np.bincount(tags), sample_num) if func is ent else 0.0

    val = []
    iv_val = []
    for f in features.T:
        tag, cnt = np.unique(f, return_counts=True)
        count = 0.0
        for t, c in zip(tag, cnt):
            sub_tags = tags[f == t]
            count += (c/sample_num) * func(np.bincount(sub_tags), c)
        if iv:
            iv_val.append(func(cnt, sample_num))
        val.append(base_ent - count if base_ent > 0.0 else count)

    return (val, iv_val) if iv else val

# ID3算法
def gain(features: np.ndarray, tags:np.ndarray) -> int:
    return int(np.argmax(val_list(features, tags, ent)))

# C4.5算法
def gain_ratio(features: np.ndarray, tags: np.ndarray) -> int:
    value, iv = val_list(features, tags, ent, True)
    if not value or not iv:
        return -1
    avg = np.mean(value)
    mask = (value <= avg)
    ratio = (np.array(value)/np.array(iv))
    ratio[mask] = 0 # 不能直接删除，否则需要算相对下标
    return int(np.argmax(ratio))

# 条件基尼系数
def cart(features: np.ndarray, tags: np.ndarray) -> int:
    return int(np.argmin(val_list(features, tags, gini)))

# 生成决策树
def create_decision_tree(features: np.ndarray, tags: np.ndarray, idx_list: list[int],
                         classifier: callable([[np.ndarray, np.ndarray], int])
                         ) -> DecisionTreeNode | None:
    if len(np.unique(tags)) == 1: # 如果预测类别只有一种，就停止决策树的生长
        return DecisionTreeNode(-1, tags[0])
    if len(idx_list) == 0: # 如果特征类别没了，没有能选择的特征，就停止决策树的生长
        max_tag = np.argmax(np.bincount(tags))
        return DecisionTreeNode(-1, max_tag)

    # 获取最佳特征下标
    idx = classifier(features, tags)
    if idx == -1:
        return None
    value = features[:, idx] # 获取特征列
    classes = np.unique(value) # 获取特征类别

    # 更新数据集
    new_features = np.delete(features, obj=idx, axis=1)
    new_idx_list = copy.deepcopy(idx_list)
    new_idx_list.pop(idx) # 删除特征列表中被选中的特征

    # 生成子节点
    children = []
    for cls in classes:
        # 划分数据集
        sub_list = (cls == value)
        sub_features = new_features[sub_list]
        sub_tags = tags[sub_list]

        child = create_decision_tree(sub_features, sub_tags, new_idx_list, classifier)
        child.val = cls
        children.append(child)

    return DecisionTreeNode(idx_list[idx], idx, children)

# 画点
def draw_circle(node: DecisionTreeNode, x: float, y: float, ax: plt.axes) -> None:
    if node.is_leaf:
        node_shape = Circle((x, y), 0.05, facecolor="#90EE90", edgecolor="black", linewidth=1.2)
        ax.add_patch(node_shape)
        ax.text(x, y, f"结果: {node.val}", ha="center", va="center", fontsize=11)
    else:
        node_shape = Rectangle(
            (x - 0.075, y - 0.05),
            0.15, 0.1,
            facecolor="#87CEFA", edgecolor="black", linewidth=1.2
        )
        ax.add_patch(node_shape)
        ax.text(x, y, f"特征: {node.feature}", ha="center", va="center", fontsize=9)

# BFS获取树高
def tree_height(root: DecisionTreeNode) -> int:
    queue = deque([root])
    level = 0
    while queue:
        sz = len(queue)
        level += 1
        for _ in range(sz):
            node = queue.popleft()
            if not node.is_leaf:
                queue.extend(node.children)

    return level

# 可视化决策树
def plot_decision_tree(root: DecisionTreeNode, max_level: int, ax: plt.axes) -> None:
    nodes = {}
    height = 1/max_level # 计算层高

    # 使用BFS遍历整个树，一次性画出节点和线
    level = 0
    queue = deque([root])
    while queue:
        level += 1
        branch_gap = 0.5 - 0.13 * level
        sz = len(queue)
        for _ in range(sz):
            node = queue.popleft()
            if level == 1:
                x = 0.5
                y = (max_level - level + 1) / max_level - 0.1
                nodes[node] = (x, y)
                draw_circle(node, x, y, ax)
            if not node.is_leaf:
                parent_x, parent_y = nodes[node]
                low = parent_x - branch_gap * (len(node.children) - 1) / 2
                for child in node.children:
                    x = low
                    low += branch_gap
                    y = parent_y - height
                    nodes[child] = (x, y)

                    con = ConnectionPatch(
                        (parent_x, parent_y - 0.05),
                        (x, y + 0.05),
                        coordsA="axes fraction", coordsB="axes fraction",
                        arrowstyle="-", color="black", linewidth=1.2
                    )
                    ax.add_artist(con)
                    ax.text((parent_x + x) / 2, (parent_y + y) / 2,
                            f"值: {child.val}",
                            ha="center", va="center", fontsize=10,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2))

                    draw_circle(child, x, y, ax)

                queue.extend(node.children)

    # 关闭坐标轴
    ax.axis("off")

# 预测数据
def predict(root: DecisionTreeNode, sample: np.ndarray) -> int | None:
    if root is None:
        return None
    if root.is_leaf:
        return root.val

    # 获取测试样本当前特征的类别，选择对应的类别递归
    genre = sample[root.feature]
    for child in root.children:
        if child.val == genre:
            return predict(child, sample)

    return None

# 获取热力图
def get_heatmap(root: DecisionTreeNode, train_tag_list: np.ndarray,
                features: np.ndarray, tags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_tag = np.unique(tags)
    tag_map = {tag:i for i, tag in enumerate(unique_tag)}
    sz = len(unique_tag)
    confusion_matrix = zeros((sz, sz))

    majority = np.argmax(np.bincount(train_tag_list))
    for feature, tag in zip(features, tags):
        result = predict(root, feature)
        if result in unique_tag: # 如果预测值在测试集的特征类别中，直接添加到混淆矩阵中
            confusion_matrix[tag_map[tag], tag_map[result]] += 1
        else: # 如果预测值不在测试集的特征列表中，使用大多数的特征填充
            confusion_matrix[tag_map[tag], tag_map[majority]] += 1
    return confusion_matrix, unique_tag

# 可视化热力图
def plot_heatmap(mat_cf: np.ndarray, unique_tag: np.ndarray, ax: plt.axes) -> None:
    # 获取类名
    class_names = list(unique_tag)
    class_names_list = range(len(class_names))

    im = ax.imshow(mat_cf)

    # 设置热力图下标标注
    ax.set_xticks(class_names_list)
    ax.set_xticklabels(labels=class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(class_names_list)
    ax.set_yticklabels(labels=class_names)

    # 给热力图上添加数据
    for x in class_names_list:
        for y in class_names_list:
            value = mat_cf[x, y]
            text_color = "black" if im.norm(value) > 0.5 else "white"
            ax.text(y, x, value, ha="center", va="center", color=text_color, fontweight="bold")

    # 添加热力条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sample Count")

# 评估并打印评估指标
def evaluate(algorithm: str, mat_cf: np.ndarray, tags: np.ndarray) -> None:
    print("************************************")
    print(f"{algorithm}算法生成的决策树模型的测试数据如下：")

    sample_num = mat_cf.shape[0]

    # 准确率
    tp_list = [mat_cf[x][x] for x in range(sample_num)]
    accuracy = np.sum(tp_list)/np.sum(mat_cf)
    print(f"模型准确率为{accuracy*100:.2f}%")

    # 精确率和召回率
    precision_list = []
    recall_list = []
    for i in range(sample_num):
        tp = tp_list[i]
        pd = np.sum(mat_cf[:][i])
        rd = np.sum(mat_cf[i])
        precision = tp/pd if pd else 0.0
        precision_list.append(precision)
        recall = tp/rd if rd else 0.0
        recall_list.append(recall)
        print(f"类别{tags[i]}的精确率为{precision*100:.2f}%，召回率为{recall*100:.2f}%")

    print(f"平均精确率为{np.mean(precision_list)*100:.2f}%")
    print(f"平均召回率为{np.mean(recall_list)*100:.2f}%")


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