import scipy.io
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']    # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来显示负号

# 读取mat文件数据
def load_mat_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    mat_data = scipy.io.loadmat(filepath)
    X = mat_data['X']
    y = mat_data['y'].ravel() # 将二维的y转化为一维的
    return X, y

# 将分布图画上去
def distribution(ax:plt.axes, x: np.ndarray, y: np.ndarray, text: str) -> None:
    ax.scatter(x[y == 0, 0], x[y == 0, 1], s=12, linewidths=0.25,
                c='lightblue', label='Class 0', edgecolors='k')
    ax.scatter(x[y == 1, 0], x[y == 1, 1], s=12, linewidths=0.25,
                c='salmon', label='Class 1', edgecolors='k')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.set_title(text)

# 原本是要做成原图和画上超平面的图一起的，但是效果不好，砍掉了
def contrast_show(text: str, svm, ax: plt.axes, X: np.ndarray, y: np.ndarray) -> None:
    # distribution(ax[0], X, y, text)

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, alpha=0.3, cmap=plt.cm.coolwarm)
    distribution(ax, X, y, text)

def uniformize(datas: np.ndarray) -> np.ndarray:
    if datas.size == 0:
        return datas
    if len(datas.shape) > 1:
        col_size = datas.shape[1]
        for col in range(col_size):
            col_val = datas[:, col]
            max_val = np.max(col_val)
            min_val = np.min(col_val)
            if np.isclose(max_val, min_val):
                datas[:, col] = 0.0
            else:
                datas[:, col] = (col_val - min_val) / (max_val - min_val)
    else:
        max_val = np.max(datas)
        min_val = np.min(datas)
        if np.isclose(max_val, min_val):
            datas = np.zeros_like(datas)
        else:
            datas = (datas - min_val) / (max_val - min_val)

    return datas

if __name__ == "__main__":
    X_train, y = load_mat_data("source/ex6data2.mat")

    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X_train))  # 用独立生成器做随机排列
    idx = np.random.permutation(len(X_train))
    train_idx = idx[:int(0.8 * len(X_train))]
    test_idx = idx[int(0.8 * len(X_train)):]
    x_train, y_train = X_train[train_idx], y[train_idx]
    x_test, y_test = X_train[test_idx], y[test_idx]

    uniformize(x_train)
    uniformize(x_test)

    # 生成线性核支持向量机
    linear_model = SVC(kernel="linear", C=1.0)
    # 生成高斯RBF核支持向量机
    gauss_model = SVC(kernel="rbf", gamma="scale", C=1.0)
    # 生成多项式核支持向量机
    poly_model = SVC(kernel="poly", gamma="auto", degree=3, C=1.0, coef0=0.0)

    models = {
        "线性核": linear_model,
        "RBF核": gauss_model,
        "多项式核": poly_model
    }

    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    for i, (name, model) in enumerate(models.items(), 0):
        model.fit(x_train, y_train)
        predict_res = model.predict(x_test)
        print(f"{name}的准确率为{accuracy_score(y_test, predict_res)*100:.1f}%")

        contrast_show(name, model, axs[i], x_train, y_train)
    plt.show()

