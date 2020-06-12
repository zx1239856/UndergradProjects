//
// Created by zx on 2020/3/25.
//

#ifndef MVC_POISSONCLONER_H
#define MVC_POISSONCLONER_H
#include "BaseCloner.h"
#include <unordered_map>
#include <vector>
#include <Eigen/Sparse>

class PoissonCloner : public BaseCloner {
public:
    PoissonCloner(cv::Mat &src, cv::Mat &dst, cv::Mat &mask);

    void startClone(int delta_x, int delta_y) override;

private:
    struct hash_pair {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2>& p) const
        {
            auto hash1 = std::hash<T1>{}(p.first);
            auto hash2 = std::hash<T2>{}(p.second);
            return hash1 ^ 0xdeadbeef + hash2;
        }
    };

    cv::Mat _mask;
    std::vector<std::pair<int, int>> _points;
    std::unordered_map<std::pair<int, int>, size_t, hash_pair> _point_index;
    Eigen::SparseMatrix<float> _coeffs;
    Eigen::VectorXf _b, _g, _r;
    Eigen::SparseLU<Eigen::SparseMatrix<float>> _solver;
    std::vector<std::pair<int, int>> _contours;

    void init();
};

#endif //MVC_POISSONCLONER_H
