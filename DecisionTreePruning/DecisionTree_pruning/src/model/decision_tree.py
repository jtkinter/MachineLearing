from src.utils.common_import import np, copy

class DecisionTreeNode:
    def __init__(self, feature, val, more_tag, children = None):
        self.feature = feature # 非叶子节点存放特征
        self.val = val # 非叶子节点存放feature的类别，叶子节点存放预测值
        self.major_tag = more_tag # 获取当前节点下最多的类
        self.children = children
        self.is_leaf = children is None

# 决策树核心代码
def core_create_tree(features: np.ndarray, tags: np.ndarray, idx_list: list[int],
                     classifier: callable([[np.ndarray, np.ndarray], int]),
                     verify_f: np.ndarray = None, verify_t: np.ndarray = None,
                     mask: np.ndarray = None, pruning = False
                     ) -> DecisionTreeNode | None:
    max_tag = int(np.argmax(np.bincount(tags)))
    if len(np.unique(tags)) == 1:  # 如果预测类别只有一种，就停止决策树的生长
        return DecisionTreeNode(-1, tags[0], max_tag)
    if len(idx_list) == 0:  # 如果特征类别没了，没有能选择的特征，就停止决策树的生长
        return DecisionTreeNode(-1, max_tag, max_tag)

    # 获取最佳特征下标
    idx = classifier(features, tags)
    if idx == -1:
        return None
    value = features[:, idx]  # 获取特征列
    classes = np.unique(value)  # 获取特征类别

    # 更新数据集
    new_features = np.delete(features, obj=idx, axis=1)
    new_idx_list = copy.deepcopy(idx_list)
    new_idx_list.pop(idx)  # 删除特征列表中被选中的特征

    # 生成子节点
    children = []
    for cls in classes:
        # 划分数据集
        sub_list = (cls == value)
        sub_features = new_features[sub_list]
        sub_tags = tags[sub_list]

        update_mask = (cls == verify_f[:, idx_list[idx]]) & mask if not mask is None else mask
        child = core_create_tree(sub_features, sub_tags, new_idx_list, classifier,
                                 verify_f, verify_t, update_mask, pruning)
        child.val = cls
        children.append(child)

    root = DecisionTreeNode(idx_list[idx], idx, max_tag, children)
    if pruning:
        from src.evaluation.evaluator import tree_accuracy

        current_ver_features = verify_f[mask]
        current_ver_tags = verify_t[mask]

        # 获取生成子树前的精度
        pre = tree_accuracy(None, max_tag, current_ver_tags)
        print(f"生成子树前的精度：{pre}")

        # 获取分裂后的精度
        mod = tree_accuracy(root, current_ver_features, current_ver_tags)
        print(f"生成子树后的精度：{mod}")

        if mod <= pre:
            return DecisionTreeNode(-1, max_tag, max_tag)

    return root

# 生成正常的决策树
def create_decision_tree(features: np.ndarray, tags: np.ndarray, idx_list: list[int],
                         classifier: callable([[np.ndarray, np.ndarray], int])
                         ) -> DecisionTreeNode | None:
    return core_create_tree(features, tags, idx_list, classifier)

# 生成预剪枝处理过的决策树
def create_decision_tree_pre_pruning(features: np.ndarray, tags: np.ndarray,
                                     idx_list: list[int], verify_f: np.ndarray, verify_t: np.ndarray,
                                     classifier: callable([[np.ndarray, np.ndarray], int]),
                                     ) -> DecisionTreeNode | None:
    mask = np.ones(len(verify_t), dtype=bool)
    return core_create_tree(features, tags, idx_list, classifier, verify_f, verify_t, mask, True)

# 后剪枝处理
def post_pruning(root: DecisionTreeNode, verify_f: np.ndarray, verify_t: np.ndarray) -> DecisionTreeNode | None:
    if root is None:
        return None
    if root.is_leaf:
        return root
    for idx, child in enumerate(root.children):
        root.children[idx] = post_pruning(child, verify_f, verify_t)

    from src.evaluation.evaluator import tree_accuracy

    post = tree_accuracy(root, verify_f, verify_t)
    node = DecisionTreeNode(-1, root.major_tag, root.major_tag)
    mod = tree_accuracy(node, verify_f, verify_t)

    return node if mod > post else root

def create_decision_tree_post_pruning(features: np.ndarray, tags: np.ndarray,
                                     idx_list: list[int], verify_f: np.ndarray, verify_t: np.ndarray,
                                     classifier: callable([[np.ndarray, np.ndarray], int]),
                                     ) -> DecisionTreeNode | None:
    root = create_decision_tree(features, tags, idx_list, classifier)
    return post_pruning(root, verify_f, verify_t)
