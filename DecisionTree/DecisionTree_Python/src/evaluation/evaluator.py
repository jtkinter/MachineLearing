from src.utils.common_import import np, zeros
from src.model.decision_tree import DecisionTreeNode

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
    tag_map = {tag: i for i, tag in enumerate(unique_tag)}
    sz = len(unique_tag)
    confusion_matrix = zeros((sz, sz))

    majority = np.argmax(np.bincount(train_tag_list))
    for feature, tag in zip(features, tags):
        result = predict(root, feature)
        if result in unique_tag:  # 如果预测值在测试集的特征类别中，直接添加到混淆矩阵中
            confusion_matrix[tag_map[tag], tag_map[result]] += 1
        else:  # 如果预测值不在测试集的特征列表中，使用大多数的特征填充
            confusion_matrix[tag_map[tag], tag_map[majority]] += 1
    return confusion_matrix, unique_tag