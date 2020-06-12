/*
 * Created by zx on 19-3-8.
 */

#ifndef MVC_MVCCLONER_H
#define MVC_MVCCLONER_H

#include <opencv2/core/core.hpp>
#include <memory>
#include "BaseCloner.h"
#include "AdaptiveMesh.h"

class MVCCloner : public BaseCloner {
public:
    MVCCloner(std::shared_ptr<mesh::AdaptiveMesh> mesh, cv::Mat &src, cv::Mat &dst);

    void startClone(int delta_x, int delta_y) override;

private:
    std::shared_ptr<mesh::AdaptiveMesh> _mesh;

    // helper methods
    static int round(double val) { return val + .5; }
    static uchar roundAndNormalize(double val) 
    { 
        if(val < 0)return 0;
        else if(val > 255)return 255;
        else return static_cast<uchar>(round(val));
    }
    static double getTan(const mesh::Point &origin, const mesh::Point &a, const mesh::Point &b);
    static double getDist(const mesh::Point &a, const mesh::Point &b);
    static uchar getPixel(const cv::Mat &img, int x, int y, int ch);
    static void setPixel(cv::Mat &img, uchar val, int x, int y, int ch);
    static std::tuple<double, double, double> getInterpolationParams(const mesh::Point &p1, double c1, const mesh::Point &p2, double c2, const mesh::Point & p3, double c3);
    static double crossProduct(const mesh::Point &p1, const mesh::Point &p2);
    static bool inTriangle(const mesh::Point &p1, const mesh::Point &p2, const mesh::Point &p3, const mesh::Point &p);
};


#endif //MVC_MVCCLONER_H
