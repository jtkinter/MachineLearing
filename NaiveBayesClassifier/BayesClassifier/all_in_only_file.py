import numpy as np

class ContinuousStats:
    def __init__(self, mean = 0.0, variance = 0.0, deviation = 0.0):
        self.mean = mean
        self.variance = variance
        self.deviation = deviation

# 导入数据
def load_data(filepath: str):
    samples = []
    with open(filepath, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line is None:
                continue
            parts = [float(x.strip()) for x in line.split(',') if x.strip()]
            if parts is None:
                continue
            samples.append(parts)
    return np.array(samples)

def stats(values: list) -> ContinuousStats | None:
    if values is None:
        return None
    avg = float(np.mean(values))
    cnt = 0.0
    for val in values:
        gap = val-avg
        cnt += gap*gap
    variance = cnt / len(values)
    deviation = variance**0.5

    return ContinuousStats(avg, variance, deviation)

# 生成概率统计表
def get_prob_cnt(data: np.ndarray, preserve: bool = False) -> (dict, dict, dict):
    ds_list = {}
    ct_list = {}

    # 分离离散特征和连续特征
    transform_data = data.T
    disperse = []
    continuous = []
    tag_t = transform_data[-1]
    for feature in transform_data[:-1]:
        flag = 0
        for f in feature:
            if f != int(f):
                flag = 1
                break
        if flag:
            continuous.append(feature)
        else:
            disperse.append(feature)

    # 计算拉普拉斯修正需要的同一特征的所有类型
    ds_vocab = []
    for ds_feature in disperse:
        vocab = len(np.unique(ds_feature))
        ds_vocab.append(vocab)

    # 计算先验概率
    total = float(len(data))
    tag_list, tag_num = np.unique(data[:,-1], return_counts=True)
    tag_num = tag_num.astype(float)
    prior_prob = {}
    for tag, num in zip(tag_list, tag_num):
        prior_prob[tag] = num / total if not preserve else np.log(num / total)

    for (tag, num) in zip(tag_list, tag_num):
        # 计算离散特征概率
        ds_prob = []
        for i in range(len(disperse)):
            ds_feature = disperse[i]
            vocab = np.unique(ds_feature)
            feature_cnt = {x:1.0 for x in vocab}
            for j in range(len(tag_t)):
                if tag_t[j] == tag:
                    feature_cnt[ds_feature[j]] += 1.0
            for (key, value) in feature_cnt.items():
                prob = value/(num+ds_vocab[i])
                feature_cnt[key] = prob if not preserve else np.log(prob)
            ds_prob.append(feature_cnt)
        ds_list[tag] = ds_prob

        # 计算连续特征概率密度
        ct_prob_dense = []
        for ct_feature in continuous:
            mask = (tag == tag_t)
            sub_ct = ct_feature[mask]
            ct_prob_dense.append(stats(sub_ct))
        ct_list[tag] = ct_prob_dense


    return prior_prob, ds_list, ct_list

# 概率密度函数
def prob_density(ct_data: ContinuousStats, test: float, preserve: bool = False):
    gap = test - ct_data.mean
    if preserve:
        den = -0.5*np.log(2*np.pi)-np.log(ct_data.deviation)
        mol = -gap*gap / (2*ct_data.variance)
        return den+mol
    else:
        return (1 / ((2 * np.pi)**0.5 * ct_data.deviation))*np.exp(-gap*gap/(2*ct_data.variance))

# 比较函数
def bayes_classifier(prior_prob: dict, ds_list: dict, ct_list: dict, test: np.ndarray, preserve: bool = False) -> int:
    disperse = []
    continuous = []
    for t in test:
        if t != int(t):
            continuous.append(t)
        else:
            disperse.append(t)

    max_label = -1
    max_value = -1e18
    for tag, prob in prior_prob.items():
        val = prob
        if preserve:
            for i in range(len(disperse)):
                val += ds_list[tag][i][disperse[i]]
            for i in range(len(continuous)):
                val += prob_density(ct_list[tag][i], continuous[i], preserve)
        else:
            for i in range(len(disperse)):
                val *= ds_list[tag][i][disperse[i]]
            for i in range(len(continuous)):
                val *= prob_density(ct_list[tag][i], continuous[i], preserve)
        print(f"类别{tag:.0f}的概率：{val}")
        if max_value < val:
            max_value = val
            max_label = tag
    return max_label

# 遍历测试集，预测标签
def predict(prior_prob: dict, ds_list: dict, ct_list: dict, testset: np.ndarray,
            preserve: bool = False, evaluate: bool = False) -> list[float]:
    results = []
    for test in testset:
        res = bayes_classifier(prior_prob, ds_list, ct_list, test, preserve)
        results.append(res)
    return results


# TODO: 拓展测试集评估模型
if __name__ == "__main__":
    train_data = load_data("dataSource/encoded_dataset.txt")
    test_data = load_data("dataSource/encoded_testset.txt")
    prevent_overflow = False
    prior, ds, ct = get_prob_cnt(train_data, prevent_overflow)
    res = predict(prior, ds, ct, test_data, prevent_overflow)
    print(res)
    print(prior)
    print(ds)
    for (key, value) in ct.items():
        print(key)
        for v in value:
            print(v.mean, v.variance, v.deviation)