#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

// 定义样本结构体：二维特征 + 标签（+1/-1）
struct Sample {
    double x1;   // 特征1
    double x2;   // 特征2
    int label;   // 标签（仅支持+1和-1）
};

// SVM类（基于SMO算法的线性SVM）
class SVM {
private:
    double C;                // 惩罚系数（越大对误分类惩罚越重）
    double tol;              // 容错率（迭代停止条件）
    int max_iter;            // 最大迭代次数
    std::vector<Sample> samples;  // 训练数据集
    std::vector<double> alphas;   // 拉格朗日乘子
    double b;                // 偏置项
    std::vector<double> w;   // 权重向量（二维）

    // 线性核函数：K(xi, xj) = xi·xj（点积）
    double kernel(const Sample& xi, const Sample& xj) {
        return xi.x1 * xj.x1 + xi.x2 * xj.x2;
    }

    // 计算样本i的预测值 f(xi) = sum(alpha_j*y_j*K(xi,xj)) + b
    double predict_value(const Sample& xi) {
        double fx = 0.0;
        for (size_t j = 0; j < samples.size(); ++j) {
            fx += alphas[j] * samples[j].label * kernel(xi, samples[j]);
        }
        fx += b;
        return fx;
    }

    // 计算样本i的误差 Ei = f(xi) - yi
    double calc_error(int i) {
        return predict_value(samples[i]) - samples[i].label;
    }

    // 选择第二个待优化的alpha（SMO核心：最大化|Ei - Ej|）
    int select_j(int i, double Ei) {
        int best_j = -1;
        double max_abs_diff = 0.0;
        double Ej;

        for (size_t j = 0; j < samples.size(); ++j) {
            if (j == i) continue;
            Ej = calc_error(j);
            double abs_diff = fabs(Ei - Ej);
            if (abs_diff > max_abs_diff) {
                max_abs_diff = abs_diff;
                best_j = j;
            }
        }
        // 若未找到则随机选一个
        if (best_j == -1) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dist(0, samples.size() - 1);
            do {
                best_j = dist(gen);
            } while (best_j == i);
        }
        return best_j;
    }

    // 裁剪alpha到[0, C]范围
    double clip_alpha(double a, double L, double H) {
        if (a < L) return L;
        if (a > H) return H;
        return a;
    }

    // 更新权重向量w（训练完成后计算）
    void update_w() {
        w.assign(2, 0.0);
        for (size_t i = 0; i < samples.size(); ++i) {
            w[0] += alphas[i] * samples[i].label * samples[i].x1;
            w[1] += alphas[i] * samples[i].label * samples[i].x2;
        }
    }

public:
    // 构造函数：初始化参数
    SVM(double c = 1.0, double tolerance = 1e-3, int max_iter = 1000)
        : C(c), tol(tolerance), max_iter(max_iter), b(0.0) {
    }

    // 训练函数（SMO核心迭代）
    void train(const std::vector<Sample>& train_samples) {
        samples = train_samples;
        int n = samples.size();
        alphas.assign(n, 0.0);  // 初始化alpha为0
        int iter = 0;

        while (iter < max_iter) {
            int alpha_pair_changed = 0;  // 记录本轮更新的alpha对数

            for (int i = 0; i < n; ++i) {
                double Ei = calc_error(i);  // 计算样本i的误差

                // 检查是否满足KKT条件（不满足则需要优化）
                bool kkt_violate = (samples[i].label * Ei < -tol && alphas[i] < C) ||
                    (samples[i].label * Ei > tol && alphas[i] > 0);

                if (kkt_violate) {
                    int j = select_j(i, Ei);  // 选择第二个优化的alpha_j
                    double Ej = calc_error(j);

                    // 保存旧的alpha值
                    double alpha_i_old = alphas[i];
                    double alpha_j_old = alphas[j];

                    // 计算alpha_j的上下界L和H
                    double L, H;
                    if (samples[i].label != samples[j].label) {
                        L = std::max(0.0, alphas[j] - alphas[i]);
                        H = std::min(C, C + alphas[j] - alphas[i]);
                    }
                    else {
                        L = std::max(0.0, alphas[i] + alphas[j] - C);
                        H = std::min(C, alphas[i] + alphas[j]);
                    }
                    if (fabs(L - H) < 1e-6) continue;  // 无优化空间

                    // 计算eta = 2K(xi,xj) - K(xi,xi) - K(xj,xj)
                    double eta = 2 * kernel(samples[i], samples[j])
                        - kernel(samples[i], samples[i])
                        - kernel(samples[j], samples[j]);
                    if (eta >= 0) continue;  // 非正定，跳过

                    // 更新alpha_j
                    alphas[j] -= samples[j].label * (Ei - Ej) / eta;
                    alphas[j] = clip_alpha(alphas[j], L, H);  // 裁剪到[L, H]

                    // 若alpha_j变化过小，跳过
                    if (fabs(alphas[j] - alpha_j_old) < 1e-6) continue;

                    // 更新alpha_i
                    alphas[i] += samples[i].label * samples[j].label * (alpha_j_old - alphas[j]);

                    // 更新偏置项b
                    double b1 = b - Ei - samples[i].label * (alphas[i] - alpha_i_old) * kernel(samples[i], samples[i])
                        - samples[j].label * (alphas[j] - alpha_j_old) * kernel(samples[i], samples[j]);
                    double b2 = b - Ej - samples[i].label * (alphas[i] - alpha_i_old) * kernel(samples[i], samples[j])
                        - samples[j].label * (alphas[j] - alpha_j_old) * kernel(samples[j], samples[j]);

                    if (0 < alphas[i] && alphas[i] < C) {
                        b = b1;

                    }
                    else if (0 < alphas[j] && alphas[j] < C) {
                        b = b2;
                    }
                    else {
                        b = (b1 + b2) / 2.0;
                    }

                    alpha_pair_changed++;  // 记录更新
                }
            }

            // 若本轮无alpha更新，迭代次数+1；否则重置迭代次数
            if (alpha_pair_changed == 0) {
                iter++;
            }
            else {
                iter = 0;
            }
        }

        update_w();  // 训练完成后计算权重w
        std::cout << "SVM训练完成！权重w: [" << w[0] << ", " << w[1] << "], 偏置b: " << b << std::endl;
    }

    // 预测函数：输入二维特征，返回标签（+1/-1）
    int predict(double x1, double x2) {
        Sample s{ x1, x2, 0 };
        double fx = predict_value(s);
        return fx > 0 ? 1 : -1;
    }

    // 获取支持向量（alpha>0的样本）
    std::vector<Sample> get_support_vectors() {
        std::vector<Sample> sv;
        for (size_t i = 0; i < samples.size(); ++i) {
            if (alphas[i] > 1e-6) {  // alpha非零即为支持向量
                sv.push_back(samples[i]);
            }
        }
        return sv;
    }
};

// 生成测试数据集：两类二维点（线性可分）
std::vector<Sample> generate_test_data(int n = 100) {
    std::vector<Sample> data;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist1(1.0, 0.5);  // 第一类：(1,1)附近
    std::normal_distribution<> dist2(5.0, 0.5);  // 第二类：(5,5)附近

    // 生成第一类（标签+1）
    for (int i = 0; i < n / 2; ++i) {
        data.push_back({ dist1(gen), dist1(gen), 1 });
    }
    // 生成第二类（标签-1）
    for (int i = 0; i < n / 2; ++i) {
        data.push_back({ dist2(gen), dist2(gen), -1 });
    }
    return data;
}

int main() {
    // 1. 生成测试数据集
    std::vector<Sample> train_data = generate_test_data(200);
    std::cout << "生成训练样本数：" << train_data.size() << std::endl;

    // 2. 初始化SVM并训练
    SVM svm(1.0, 1e-3, 1000);  // C=1, 容错率1e-3, 最大迭代1000
    svm.train(train_data);

    // 3. 获取支持向量
    std::vector<Sample> sv = svm.get_support_vectors();
    std::cout << "支持向量数量：" << sv.size() << std::endl;

    // 4. 测试预测
    std::vector<Sample> test_samples = {
        {1.2, 0.8, 1},    // 第一类
        {0.9, 1.1, 1},    // 第一类
        {5.1, 4.9, -1},   // 第二类
        {4.8, 5.2, -1},   // 第二类
        {3.0, 3.0, 0},    // 边界点（测试）
        {0.5, 5.0, 0}     // 混合点（测试）
    };

    std::cout << "\n预测结果：" << std::endl;
    for (const auto& s : test_samples) {
        int pred = svm.predict(s.x1, s.x2);
        std::cout << "样本(" << s.x1 << ", " << s.x2 << ")：真实标签=" << s.label
            << "，预测标签=" << pred << std::endl;
    }

    return 0;
}