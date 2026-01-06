#include "datadeal.h"
#include "plotshow.h"
#include "svm.h"
#include "evaluate.h"

// 手动补充matio 1.5.29缺失的宏
//#ifndef MAT_FIND_BY_NAME
//#define MAT_FIND_BY_NAME 0
//#endif
//#ifndef MAT_FIND_BY_INDEX
//#define MAT_FIND_BY_INDEX 1
//#endif

int main(int argc, char** argv) {

    const std::string trainfile = "source/ex6data2.mat";
    const std::string testfile = "source/ex6data1.mat";


    auto samples = loadData(trainfile);
    auto stats = NormalizeStats();
    uniformize(samples);
    //stats =  normalize(samples, stats);

    auto classifier = group(samples);
    distrshow(classifier);

    auto model = build_model(samples, KernelType::POLYNOMIAL);
    std::cout << "alpha一览：" << std::endl;
    for (double a : model.alpha)
    {
        std::cout << a << " ";
    }
    std::cout << std::endl;

    SVM svm_model(model);
    auto class_support_group = group(svm_model.support_vectors);
    supportshow(class_support_group);

    auto testset = loadData(testfile);
    uniformize(testset);
    //stats = normalize(testset, stats);
    evaluate(svm_model, testset);

    return 0;
}