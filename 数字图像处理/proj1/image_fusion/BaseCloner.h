//
// Created by zx on 2020/3/25.
//

#ifndef MVC_BASECLONER_H
#define MVC_BASECLONER_H
#include <opencv2/core/core.hpp>


class BaseCloner {
public:
    BaseCloner(cv::Mat &src, cv::Mat &dst) : _src(src), _dst(dst), _output(dst.clone()) {}

    virtual void startClone(int delta_x, int delta_y) = 0;

    cv::Mat getResult() { return _output; }

    virtual ~BaseCloner() = default;

protected:
    cv::Mat _src;
    cv::Mat _dst;
    cv::Mat _output;
};

#endif //MVC_BASECLONER_H
