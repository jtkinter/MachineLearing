from src.utils.common_import import np, copy

class DecisionTreeNode:
    def __init__(self, feature, val, children = None):
        self.feature = feature # 非叶子节点存放特征
        self.val = val # 非叶子节点存放feature的类别，叶子节点存放预测值
        self.children = children
        self.is_leaf = children is None

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
