from src.utils.common_import import np

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
    return int(np.argmax(ratio if len(ratio) > 0 else value))

# 条件基尼系数
def cart(features: np.ndarray, tags: np.ndarray) -> int:
    return int(np.argmin(val_list(features, tags, gini)))