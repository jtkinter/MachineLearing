import numpy as np
import matplotlib.pyplot as plt
import time

tag_map = {
    "largeDoses": 0,
    "smallDoses": 1,
    "didntLike": 2
}

# 导入数据
def load_data(filepath):
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

    ranges = max_val-min_val
    ranges[ranges < 1e-9] = 1.0

    return (features-min_val)/ranges

# 计算欧式距离
def euclidean_dist(test, train):
    return np.sum((test - train) ** 2, axis=1)

# 计算各类标签的概率
def predict_prob(test, train_data, k):
    # dist = []
    # for t in train_data:
    #     dist.append([euclidean_dist(test, t[:3]), t[3]])
    # dist = sorted(dist, key=lambda x: x[0])
    # prob = np.zeros(3, dtype=np.int32)
    # for i in range(k):
    #     prob[int(dist[i][1])] += 1
    # return prob
    train_features = train_data[:,:3]
    dist = euclidean_dist(test, train_features)
    k_indices = np.argpartition(dist, k)[:k] # 完成一次快速排序
    k_tags = train_data[k_indices, 3].astype(int)
    prob = np.bincount(k_tags, minlength=3) / k
    return prob

# 选择最大概率作为当前测试样本的预测类型
def predict_type(test, train_data, k):
    return max(enumerate(predict_prob(test, train_data, k)), key=lambda x:x[1])[0]

# 使用k折交叉验证，计算准确率和三分类特征混淆矩阵
def k_folds_cross_valid_acc(features: np.ndarray, k, k_fold):
    fold_size = int(len(features)/k_fold)
    fold_accuracies = 0.0
    confusion_mat = np.zeros([3,3], dtype=np.int32)

    for i in range(k_fold):
        start = i*fold_size
        end = start+fold_size if i != k_fold-1 else len(features)
        train_data = np.concatenate([features[:start], features[end:]])
        correct = 0
        test_data = features[start:end]
        for t in test_data:
            pred_type = predict_type(t[:3], train_data, k)
            if pred_type == t[3]:
                correct += 1
            confusion_mat[int(t[3])][pred_type] += 1
        fold_accuracies += correct/fold_size
    return fold_accuracies/k_fold, confusion_mat

# 计算模型roc曲线的单个点
def count_roc(features: np.ndarray, k_fold, k_neighbor, confidence, genre):
    fold_size = int(len(features)/k_fold)
    tp, tn, fp, fn = 0, 0, 0, 0

    for i in range(k_fold):
        start = i*fold_size
        end = start+fold_size if i != k_fold-1 else len(features)
        train_data = np.concatenate([features[:start], features[end:]])
        test_data = features[start:end]
        for t in test_data:
            prob = predict_prob(t[:3], train_data, k_neighbor)
            if genre == t[3] and prob[genre] >= confidence:
                tp += 1
            elif genre != t[3] and prob[genre] < confidence:
                tn += 1
            elif genre != t[3] and prob[genre] >= confidence:
                fp += 1
            elif genre == t[3] and prob[genre] < confidence:
                fn += 1

    tpr = tp / (tp+fn) if (tp+fn) else 0.0
    fpr = fp / (fp+tn) if (fp+tn) else 0.0

    return fpr, tpr

# 整理并补全roc曲线的两个点集
def get_roc(features: np.ndarray, k_fold, k_neighbor, genre):
    confidence = np.arange(1.0, 0, -0.05)
    fpr_list = list()
    tpr_list = list()
    for conf in confidence:
        fpr, tpr = count_roc(features, k_fold, k_neighbor, conf, genre)
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    if fpr_list[0] != 0.0:
        fpr_list.insert(0, 0.0)
        tpr_list.insert(0, 0.0)
    if fpr_list[len(fpr_list)-1] != 1.0:
        fpr_list.append(1.0)
        tpr_list.append(1.0)

    return [fpr_list, tpr_list]

# 获取AUC
def get_auc(result):
    auc = 0.0
    fpr_list, tpr_list = result
    if len(fpr_list) < 2:
        return 0.0
    for i in range(len(fpr_list)-1):
        # 梯形法算AUC面积
        auc += (fpr_list[i+1]-fpr_list[i])*(tpr_list[i]+tpr_list[i+1])/2
    return auc

# 通过值寻找键列表
def get_value(dictionary: dict, target_value: int):
    return [key for key, value in dictionary.items() if value == target_value]

# 绘制roc曲线
def plot_roc(results, ax):
    class_names = list(tag_map.keys())
    for i in range(len(results)):
        ax.plot(results[i][0], results[i][1], linewidth=2, label=f"{class_names[i]} AUC: {results[i][2]:0.5f}")
    ax.plot((0, 1), (0, 1), "--", linewidth=1, label="predicted line")

    ax.legend(loc="lower right")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

# 绘制热力图
def plot_heatmap(heat_conf, ax):
    class_names = list(tag_map.keys())
    im = ax.imshow(heat_conf)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(labels=tag_map.keys(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(labels=tag_map.keys())
    ax.set_title("Three Distribution")

    for i in range(len(tag_map)):
        for j in range(len(tag_map)):
            value = heat_conf[i, j]
            # 提高可视化观感
            text_color = "black" if im.norm(value) > 0.5 else "white"
            ax.text(j, i, value, ha="center", va="center", color=text_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Sample Count")


if __name__ == "__main__":
    start = time.time()
    # 导入数据
    dataset = load_data("datingTestSet.txt")
    # 检查数据是否导入
    if not dataset:
        print("没有有效数据")
        exit()

    # 创建numpy数组
    feature_list = np.array([d["feature"] for d in dataset], dtype=np.float64)
    tag_list = np.array([d["tag"] for d in dataset], dtype=np.int32)

    # 归一化数组
    normalized_feature = normalized(feature_list)

    # 将特征和标签拼接
    tag_list = tag_list.reshape(-1, 1) # 将numpy数组转置
    feature_tag = np.hstack((normalized_feature, tag_list))

    # 洗牌
    np.random.seed(42)
    np.random.shuffle(feature_tag)

    # 选择最大准确率，选择最佳k值，并计算出最佳k值下的三分类特征混淆矩阵
    best_k = 0
    best_acc = 0
    best_confusion = np.zeros([3, 3], dtype=np.int32)
    for i in range(3, int(np.sqrt(len(feature_tag))), 2):
        acc, confusion = k_folds_cross_valid_acc(feature_tag, i, 10)
        if best_acc < acc:
            best_k = i
            best_acc = acc
            best_confusion = confusion
    print(f"十折交叉验证的最佳k值为：{best_k}，对应的准确率为：{best_acc:0.5f}")

    # 评估模型：获取ROC曲线和AUC值
    results = list()
    for i in range(3):
        res = get_roc(feature_tag, 10, best_k, i)
        auc = get_auc(res)
        results.append([*res, auc])

    end = time.time()
    print("使用时间: ", end - start)

    # 评估数据可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot_roc(results, ax1)
    plot_heatmap(best_confusion, ax2)

    plt.tight_layout()
    plt.show()