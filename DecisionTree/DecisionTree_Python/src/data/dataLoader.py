from src.utils.common_import import np

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