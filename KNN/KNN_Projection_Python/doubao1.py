import numpy as np


# 导入数据
def load_data(filepath):
    tag_map = {
        "largeDoses": 0,
        "smallDoses": 1,
        "didntLike": 2
    }

    # 读取数据
    data = []
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                print(f"特征数据无法转换：{line}")
                continue
            tag = parts[3]
            data.append({
                "feature": [parts[0], parts[1], parts[2]],
                "tag": tag_map[tag]
            })

    return data


# 归一化 利用numpy的广播机制
def normalized(features: np.ndarray):
    max_val = features.max(axis=0)
    min_val = features.min(axis=0)

    ranges = max_val - min_val
    ranges[ranges < 1e-9] = 1.0

    return (features - min_val) / ranges


def split_data(features: np.ndarray):
    split_num = int(0.8 * len(features))
    tags = features[:split_num]
    return [features[split_num:], tags[:, :3], tags[:, 3].astype(np.int32)]


# K近邻分类器
def knn_classify(test_data, train_data, train_labels, n_neighbors=3):
    """
    K近邻分类器
    """
    # 计算测试数据与所有训练数据的欧氏距离
    distances = np.sqrt(np.sum((train_data - test_data[:, np.newaxis]) ** 2, axis=2))

    # 对距离进行排序并获取前n_neighbors个最近邻居的索引
    sorted_indices = np.argsort(distances, axis=1)
    k_nearest_indices = sorted_indices[:, :n_neighbors]

    # 获取前n_neighbors个最近邻居的标签
    k_nearest_labels = train_labels[k_nearest_indices]

    # 对每个测试样本进行投票
    predictions = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=1,
        arr=k_nearest_labels
    )

    return predictions


# 计算准确率
def calculate_accuracy(predictions, true_labels):
    """
    计算分类准确率
    """
    return np.mean(predictions == true_labels)


# K折交叉验证
def k_fold_cross_validation(data, k=5, classifier=knn_classify, **kwargs):
    """
    K折交叉验证

    参数:
    data: 包含特征和标签的numpy数组，最后一列是标签
    k: 折数，默认为5
    classifier: 分类器函数，默认为knn_classify
    **kwargs: 分类器的额外参数

    返回:
    average_accuracy: 平均准确率
    fold_accuracies: 每一折的准确率列表
    """
    # 确保数据被正确洗牌
    np.random.seed(42)
    np.random.shuffle(data)

    # 计算每一折的大小
    fold_size = len(data) // k
    fold_accuracies = []

    print(f"开始{str(k)}折交叉验证...")

    for i in range(k):
        print(f"\n第{i + 1}折:")

        # 划分训练集和验证集
        # 验证集
        val_start = i * fold_size
        val_end = val_start + fold_size
        val_data = data[val_start:val_end]
        val_features = val_data[:, :-1]
        val_labels = val_data[:, -1].astype(np.int32)

        # 训练集
        train_data = np.concatenate([data[:val_start], data[val_end:]])
        train_features = train_data[:, :-1]
        train_labels = train_data[:, -1].astype(np.int32)

        print(f"  训练集大小: {len(train_data)}, 验证集大小: {len(val_data)}")

        # 使用分类器进行预测
        predictions = classifier(val_features, train_features, train_labels, **kwargs)

        # 计算准确率
        accuracy = calculate_accuracy(predictions, val_labels)
        fold_accuracies.append(accuracy)

        print(f"  准确率: {accuracy:.4f}")

    # 计算平均准确率
    average_accuracy = np.mean(fold_accuracies)
    print(f"\n{str(k)}折交叉验证结果:")
    print(f"各折准确率: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    print(f"平均准确率: {average_accuracy:.4f}")
    print(f"标准差: {np.std(fold_accuracies):.4f}")

    return average_accuracy, fold_accuracies


# 网格搜索调优
def grid_search(data, param_grid, k=5):
    """
    网格搜索调优参数

    参数:
    data: 数据集
    param_grid: 参数网格，字典格式
    k: 交叉验证折数

    返回:
    best_params: 最佳参数组合
    best_score: 最佳得分
    """
    from itertools import product

    # 获取所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    best_score = -1
    best_params = None

    print(f"开始网格搜索，共{len(param_combinations)}个参数组合...")

    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_names, params))
        print(f"\n参数组合 {i + 1}/{len(param_combinations)}: {param_dict}")

        # 执行交叉验证
        score, _ = k_fold_cross_validation(data, k=k, **param_dict)

        # 更新最佳参数
        if score > best_score:
            best_score = score
            best_params = param_dict
            print(f"  找到更好的参数组合，准确率: {score:.4f}")
        else:
            print(f"  准确率: {score:.4f}")

    print(f"\n网格搜索完成！")
    print(f"最佳参数组合: {best_params}")
    print(f"最佳准确率: {best_score:.4f}")

    return best_params, best_score


if __name__ == "__main__":
    # 加载数据
    dataset = load_data("datingTestSet.txt")

    if not dataset:
        print("没有有效数据")
        exit()

    # 准备数据
    feature_list = np.array([d["feature"] for d in dataset], dtype=np.float64)
    tag_list = np.array([d["tag"] for d in dataset], dtype=np.int32)

    # 归一化特征
    normalized_feature = normalized(feature_list)

    # 合并特征和标签
    tag_list = tag_list.reshape(-1, 1)
    feature_tag = np.hstack((normalized_feature, tag_list))

    print(f"数据集大小: {len(feature_tag)}")
    print(f"特征数量: {feature_tag.shape[1] - 1}")
    print(f"类别数量: {len(np.unique(feature_tag[:, -1]))}")
    print(f"类别分布: {np.bincount(feature_tag[:, -1].astype(int))}")

    print("\n" + "=" * 60)

    # 1. 执行5折交叉验证
    print("1. 5折交叉验证 (n_neighbors=3):")
    average_acc, fold_accs = k_fold_cross_validation(feature_tag, k=5, n_neighbors=3)

    print("\n" + "=" * 60)

    # 2. 执行10折交叉验证
    print("2. 10折交叉验证 (n_neighbors=3):")
    average_acc_10, fold_accs_10 = k_fold_cross_validation(feature_tag, k=10, n_neighbors=3)

    print("\n" + "=" * 60)

    # 3. 使用不同的n_neighbors值进行交叉验证
    print("3. 测试不同的n_neighbors值:")
    n_neighbors_values = [1, 3, 5, 7, 9]
    neighbor_accuracies = []

    for n in n_neighbors_values:
        print(f"\n使用n_neighbors={n}:")
        avg_acc, _ = k_fold_cross_validation(feature_tag, k=5, n_neighbors=n)
        neighbor_accuracies.append(avg_acc)

    best_n = n_neighbors_values[np.argmax(neighbor_accuracies)]
    print(f"\n最佳n_neighbors值: {best_n}, 最佳准确率: {max(neighbor_accuracies):.4f}")

    print("\n" + "=" * 60)

    # 4. 使用网格搜索调优参数
    print("4. 网格搜索调优:")
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9],
        'classifier': [knn_classify]  # 可以添加其他分类器
    }

    best_params, best_score = grid_search(feature_tag, param_grid, k=5)
