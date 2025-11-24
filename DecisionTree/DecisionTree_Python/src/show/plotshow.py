from src.model.decision_tree import DecisionTreeNode
from src.utils.common_import import plt, Rectangle, Circle, ConnectionPatch, deque, np

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