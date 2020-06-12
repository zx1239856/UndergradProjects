//
// Created by zx on 2020/3/25.
//
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "PoissonCloner.h"

PoissonCloner::PoissonCloner(cv::Mat &src, cv::Mat &dst, cv::Mat &mask) :
        BaseCloner(src, dst), _mask(mask) {
    CHECK_EQ(_src.type(), _dst.type()) << "Source and destination image type mismatch";
    CHECK_EQ(src.size(), mask.size()) << "Source and mask should have the same size";
    auto type = _src.type();
    CHECK(type == CV_8UC3)
                    << "Only 8U_C3 type image is supported";
    cv::threshold(_mask, _mask, 254, 255, 3);
    init();
}

inline uchar getPixel(const cv::Mat &img, int x, int y, int ch) {
    if (x < 0 || x >= img.cols)
        return 0;
    if (y < 0 || y >= img.rows)
        return 0;
    return img.at<cv::Vec3b>(y, x)[ch];
}

inline cv::Vec3b getPixel(const cv::Mat &img, int x, int y) {
    if (x < 0 || x >= img.cols)
        return {0, 0, 0};
    if (y < 0 || y >= img.rows)
        return {0, 0, 0};
    return img.at<cv::Vec3b>(y, x);
}

inline void setPixel(cv::Mat &img, uchar val, int x, int y, int ch) {
    if (x < 0 || x >= img.cols)
        return;
    if (y < 0 || y >= img.rows)
        return;
    img.at<cv::Vec3b>(y, x)[ch] = val;
}

inline uchar clip(int val) {
    return val < 0 ? 0 : (val >= 255 ? 255 : val);
}

void PoissonCloner::init() {
    using namespace cv;
    using namespace std;

    Mat area_coord;
    findNonZero(_mask, area_coord);
    auto N = area_coord.rows;
    _points.reserve(N);
    for (int i = 0; i < N; ++i) {
        auto point = make_pair(area_coord.at<int>(i, 0), area_coord.at<int>(i, 1));
        _points.emplace_back(point);
        _point_index[point] = i;
    }

    // initialize coefficient matrix and constants
    vector<Eigen::Triplet<float>> coeffs;
    coeffs.reserve(N);
    Eigen::VectorXf b(N), g(N), r(N);
    for (int i = 0; i < N; ++i) {
        auto point = _points[i];
        std::pair<int, int> neighbors[] = {{point.first - 1, point.second},
                                           {point.first + 1, point.second},
                                           {point.first,     point.second - 1},
                                           {point.first,     point.second + 1}};
        int cnt = 0;
        Vec3b lap[] = {getPixel(_src, point.first, point.second), getPixel(_src, point.first - 1, point.second),
                       getPixel(_src, point.first + 1, point.second), getPixel(_src, point.first, point.second - 1),
                       getPixel(_src, point.first, point.second + 1)};
        b(i) = 4 * (int)lap[0][0] - (int)lap[1][0] - (int)lap[2][0] - (int)lap[3][0] - (int)lap[4][0];
        g(i) = 4 * (int)lap[0][1] - (int)lap[1][1] - (int)lap[2][1] - (int)lap[3][1] - (int)lap[4][1];
        r(i) = 4 * (int)lap[0][2] - (int)lap[1][2] - (int)lap[2][2] - (int)lap[3][2] - (int)lap[4][2];
        coeffs.emplace_back(i, i, 4);
        for (const auto &p : neighbors) {
            if (_point_index.find(p) != _point_index.end()) {
                ++cnt;
                coeffs.emplace_back(i, _point_index[p], -1);
            }
        }
        if (cnt != 4) {
            _contours.emplace_back(point);
        }
    }
    _coeffs.resize(N, N);
    _coeffs.setFromTriplets(coeffs.begin(), coeffs.end());
    _solver.compute(_coeffs);
    _b = std::move(b);
    _g = std::move(g);
    _r = std::move(r);
    LOG(INFO) << "Poisson init done";
}

void PoissonCloner::startClone(int delta_x, int delta_y) {
    using namespace Eigen;
    using namespace std;
    auto N = _points.size();
    VectorXf b = _b, g = _g, r = _r;
    for (auto &it : _contours) {
        auto idx = _point_index[it];
        std::pair<int, int> neighbors[] = {{it.first - 1,it.second},
                                           {it.first + 1, it.second},
                                           {it.first,     it.second - 1},
                                           {it.first,     it.second + 1}};
        for(auto & neighbor : neighbors) {
            if(_mask.at<uchar>(neighbor.second, neighbor.first) == 0) {
                b(idx) += 1. * getPixel(_dst, neighbor.first + delta_x, neighbor.second + delta_y, 0);
                g(idx) += 1. * getPixel(_dst, neighbor.first + delta_x, neighbor.second + delta_y, 1);
                r(idx) += 1. * getPixel(_dst, neighbor.first + delta_x, neighbor.second + delta_y, 2);
            }
        }
    }
    VectorXf b_ = _solver.solve(b);
    VectorXf g_ = _solver.solve(g);
    VectorXf r_ = _solver.solve(r);

    _output = _dst.clone();
    for (int i = 0; i < N; ++i) {
        auto pp = _points[i];
        setPixel(_output, clip(round(b_(i))), pp.first + delta_x, pp.second + delta_y, 0);
        setPixel(_output, clip(round(g_(i))), pp.first + delta_x, pp.second + delta_y, 1);
        setPixel(_output, clip(round(r_(i))), pp.first + delta_x, pp.second + delta_y, 2);
    }
}