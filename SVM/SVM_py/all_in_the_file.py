import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt

kernel_type = {
    "linear": 0,
    "rbf": 1,
    "poly": 2
}

class Model:
    def __init__(self, X: np.ndarray, y: np.ndarray, kernel: int|str = "linear",
                 C: float = 1.0, tolerant:float = 1e-6, val: float = 1.0,
                 max_it: int = 50, max_no_change: int = 15, info: bool = False):
        # 训练集
        self.X = X # 特征
        self.y = y # 标签
        self.errors = np.zeros(len(X), dtype=float) # 降低时间复杂度

        # 超参数
        self.kernel = kernel # 核类型
        self.C = C # 惩罚系数
        self.tol = tolerant # 容忍度
        self.val = val # 核相关
        self.max_iteration = max_it # 最大迭代次数
        self.max_continue_no_update = max_no_change # 最大连续不更新迭代次数 -> 认为达到模型最佳，不可能再更新了，提前结束训练

        # 超平面参数
        self.w = np.zeros(X.shape[1], dtype=float) # 线性核参数
        self.alpha = np.zeros(len(X), dtype=float) # 非线性核参数
        self.b = 0.0 # 偏置

        self.info = info
        self._bind_kernel()

        self.train()

    # linear
    def linear(self, x1, x2):
        return float(np.dot(x1, x2))

    def gauss(self, x1, x2):
        dist = np.sum((x1 - x2) ** 2)
        return float(np.exp(-dist / (2 * self.val * self.val)))

    def poly(self, x1, x2):
        return float((np.dot(x1, x2) + 1) ** self.val)

    # 核函数
    def _bind_kernel(self):
        if self.kernel == "linear" or self.kernel == 0:
            self.kernel_calculate = self.linear
        elif self.kernel == "rbf" or self.kernel == 1:
            self.kernel_calculate = self.gauss
        elif self.kernel == "poly" or self.kernel == 2:
            self.kernel_calculate = self.poly
        else:
            print("kernel_calculate: 没有对应的核，使用线性核处理")
            self.kernel_calculate = self.linear

    # 计算误差
    def calculate_error(self, idx: int) -> float:
        fx = self.b
        x = self.X[idx]
        fx += sum([self.alpha[i] * self.y[i] * self.kernel_calculate(x, self.X[i]) for i in range(len(self.X))])
        return float(fx - self.y[idx])

    def update_w(self):
        self.w = np.dot(self.alpha * self.y, self.X)

    # 更新误差向量
    def update_error(self):
        for i in range(len(self.errors)):
            self.errors[i] = self.calculate_error(i)

    # 通过启发式策略，根据i寻找j
    def find_other(self, idx: int, e1: float) -> int:
        best_j = idx
        max_diff = 0.0
        for j in range(len(self.alpha)):
            if j == idx:
                continue
            e2 = self.errors[j]
            diff = np.fabs(e2-e1)
            if diff > max_diff:
                max_diff = diff
                best_j = j

        while True:
            if best_j != idx:
                break
            print("Model.find_other: 根据启发式策略，没有找到合适的j")
            best_j = random.randint(0, len(self.alpha) - 1)

        return best_j

    # 规范alpha_j的范围
    def clip(self, low: float, high: float, val: float) -> float:
        if low > high:
            tmp = low
            low = high
            high = tmp
        if val < low:
            return low
        if val > high:
            return high
        return val

    # 使用SMO算法训练
    def train(self):
        it = 0
        continue_no_update = 0
        self.update_error()
        while it < self.max_iteration and continue_no_update < self.max_continue_no_update:
            change_alpha = 0
            for i in range(len(self.alpha)):
                e1 = self.errors[i]
                a1 = self.alpha[i]
                y1 = self.y[i]
                fx1 = e1 + y1
                cases = y1 * fx1

                kkt = self.tol < a1 < self.C - self.tol or (a1 <= self.tol and cases < 1.0 - self.tol) or (a1 >= self.C - self.tol and cases > 1.0 + self.tol)
                if not kkt:
                    continue

                j = self.find_other(i, float(e1))
                x1 = self.X[i]
                x2 = self.X[j]

                e2 = self.errors[j]
                a2 = self.alpha[j]
                y2 = self.y[j]

                l, h = 0.0, 0.0
                if y1 == y2:
                    l = max(0.0, float(a2 + a1 - self.C))
                    h = min(self.C, float(a1 + a2))
                else:
                    l = max(0.0, float(a2 - a1))
                    h = min(self.C, float(self.C + a2 - a1))
                if np.fabs(l-h) < self.tol:
                    continue

                k11 = self.kernel_calculate(x1, x1)
                k12 = self.kernel_calculate(x1, x2)
                k22 = self.kernel_calculate(x2, x2)

                eta = k11 + k22 - 2 * k12
                if eta < self.tol:
                    continue

                self.alpha[j] += y2 * (e1 - e2) / eta
                self.alpha[j] = self.clip(l, h, float(self.alpha[j]))
                if np.fabs(a2-self.alpha[j]) < self.tol:
                    continue

                self.alpha[i] += y1 * y2 * (a2-self.alpha[j])

                b1 = self.b - e1 - y1 * (self.alpha[i] - a1) * k11 - y2 * (self.alpha[j] - a2) * k12
                b2 = self.b - e2 - y2 * (self.alpha[j] - a2) * k22 - y1 * (self.alpha[j] - a2) * k12

                if self.tol < self.alpha[i] < self.C - self.tol:
                    self.b = b1
                elif self.tol < self.alpha[j] < self.C - self.tol:
                    self.b = b2
                else:
                    self.b = (b1+b2)/2

                if self.info:
                    print(f"更新alpha对：({i},{j}), b = {self.b}")

                self.errors[i] = self.calculate_error(i)
                self.errors[j] = self.calculate_error(j)

                change_alpha += 1

            it += 1
            if change_alpha:
                continue_no_update = 0
                if self.info:
                    print(f"第{it}轮迭代，更新了{change_alpha*2}个alpha值...")
                if self.kernel == "linear" or self.kernel == 0:
                    self.update_w()
                    if self.info:
                        print(f"更新的w: {self.w}")
            else:
                continue_no_update += 1
                if self.info:
                    print(f"第{it}轮迭代，已经连续没有更新{continue_no_update}轮了......")

class SVM:
    def __init__(self, train_model: Model, valid_val: float = 1e-8):
        self.valid = valid_val
        self._extract_support_vector(train_model)
        self._extract_valid_arg(train_model)
        self._bind_kernel()

    # linear
    def linear(self, x1, x2):
        return float(np.dot(x1, x2))

    def gauss(self, x1, x2):
        dist = np.sum((x1 - x2) ** 2)
        return float(np.exp(-dist / (2 * self.val * self.val)))

    def poly(self, x1, x2):
        return float((np.dot(x1, x2) + 1) ** self.val)

    # 核函数
    def _bind_kernel(self):
        if self.kernel == "linear" or self.kernel == 0:
            self.kernel_calculate = self.linear
        elif self.kernel == "rbf" or self.kernel == 1:
            self.kernel_calculate = self.gauss
        elif self.kernel == "poly" or self.kernel == 2:
            self.kernel_calculate = self.poly

    def _extract_support_vector(self, train_model: Model):
        support_mask = train_model.alpha > self.valid
        self.support_X = train_model.X[support_mask].astype(np.float64)
        self.support_y = train_model.y[support_mask].astype(np.float64)
        self.alpha = train_model.alpha[support_mask].astype(np.float64)
        self.n_support = len(self.alpha)
        print(f"提取到{self.n_support}个有效支持向量")

    def _extract_valid_arg(self, train_model: Model):
        self.tol = train_model.tol
        self.b = train_model.b
        self.w = train_model.w
        self.kernel = train_model.kernel
        self.val = train_model.val

    def predict(self, x):
        cnt = 0.0
        if self.kernel == "linear":
            cnt += self.kernel_calculate(x, self.w) + self.b
        else:
            for alpha, X, y in zip(self.alpha, self.support_X, self.support_y):
                 cnt += alpha * y * self.kernel_calculate(X, x)

        if np.fabs(cnt) > self.tol:
            return 1.0 if cnt > 0.0 else -1.0
        else:
            return 0.0


    def evaluate(self, tests_x: np.ndarray, tests_y: np.ndarray):
        """
        目前实现了准确率，
        TODO：还有精确率核召回率等模型评估可以实现
        :param tests_x: 测试集特征
        :param tests_y: 测试集标签
        :return: 准确率
        """
        correct = 0
        for x, y in zip(tests_x, tests_y):
            res = self.predict(x)
            if res == y:
                correct += 1
        return correct/tests_x.shape[0]


# 将分布图画上去
def distribution(x: np.ndarray, y: np.ndarray, text: str, mark='o', s=12, linewidth = 0.25) -> None:
    plt.scatter(x[y == 0, 0], x[y == 0, 1], s=s, linewidths=linewidth, marker=mark,
               c='lightblue', label='Class 0', edgecolors='k')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], s=s, linewidths=linewidth, marker=mark,
               c='salmon', label='Class 1', edgecolors='k')


    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title(text)

# 导入数据
def load_mat_data(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    mat_data = scipy.io.loadmat(filepath)
    return mat_data['X'], mat_data['y'].ravel()

if __name__ == "__main__":
    x_train, y_train = load_mat_data("source/ex6data2.mat")
    x_test, y_test = load_mat_data("source/ex6data1.mat")

    for method in ["linear", "rbf", "poly"]:
        model = Model(x_train, y_train, method)
        svm = SVM(model)

        print(f"基于{method}核的模型准确率：{svm.evaluate(x_test, y_test)*100:.1f}%")

        distribution(x_train, y_train, method)
        distribution(svm.support_X, svm.support_y, method, '*', 16, 0.5)
        plt.show()