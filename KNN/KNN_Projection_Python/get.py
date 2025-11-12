import numpy as np
import matplotlib.pyplot as plt
import main
dataset = main.load_data("datingTestSet.txt")
# 检查数据是否导入
if not dataset:
    print("没有有效数据")
    exit()

# 创建numpy数组
feature_list = np.array([d["feature"] for d in dataset], dtype=np.float64)
tag_list = np.array([d["tag"] for d in dataset], dtype=np.int32)

# 归一化数组
normalized_feature = main.normalized(feature_list)

# 将特征和标签拼接
tag_list = tag_list.reshape(-1, 1) # 将numpy数组转置
feature_tag = np.hstack((normalized_feature, tag_list))

# 洗牌
np.random.seed(42)
np.random.shuffle(feature_tag)

acc_list: list[float] = list()
for i in range(3, int(np.sqrt(len(feature_tag))), 2):
    acc, confusion = main.k_folds_cross_valid_acc(feature_tag, i, 10)
    acc_list.append(acc)

plt.plot(range(3, int(np.sqrt(len(feature_tag))), 2), acc_list)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("K-Accuracy Curve")
plt.show()