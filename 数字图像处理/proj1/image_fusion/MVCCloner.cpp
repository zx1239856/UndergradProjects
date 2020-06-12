/*
 * Created by zx on 19-3-8.
 */

#include "MVCCloner.h"
#include <glog/logging.h>
#include <opencv2/imgproc.hpp>
#include <Eigen/Dense>

MVCCloner::MVCCloner(std::shared_ptr<mesh::AdaptiveMesh> mesh, cv::Mat &src, cv::Mat &dst) :
        BaseCloner(src, dst), _mesh(mesh) {
    CHECK_EQ(_src.type(), _dst.type()) << "Source and destination image type mismatch";
    auto type = _src.type();
    CHECK(type == CV_8UC1 || type == CV_8UC2 || type == CV_8UC3 || type == CV_8UC4)
    << "Only 8U_CX type image is supported";
}

// clone src to dst with position (delta_x, delta_y)
void MVCCloner::startClone(int delta_x, int delta_y) {
    _output = _dst.clone();
    std::vector<mesh::Point> mesh_pnt = _mesh->getPoints();
    std::vector<mesh::Triangle > mesh_triangle = _mesh->getTriangles();
    std::map<mesh::Point, int> pnt_id = _mesh->getPointId();
    std::vector<mesh::Point>  boundary_pnt = _mesh->getBoundaryPoints();
    std::map<mesh::Point, int> boundary_pnt_id;
    int channel = _src.channels();
    int bsz = boundary_pnt.size();
    for (int i = 0; i < bsz; ++i)
        boundary_pnt_id[boundary_pnt[i]] = i;

    std::vector<std::vector<double>> boundary_dif_color(bsz, std::vector<double>(channel, 0));
    std::vector<std::vector<double>> inner_dif_color(mesh_pnt.size(), std::vector<double>(channel, 0));

    // process boundary, val = f - g
    for (int i = 0; i < bsz; ++i) {
            int sx = round(boundary_pnt[i].x()), sy = round(boundary_pnt[i].y());
            int tx = sx + delta_x, ty = sy + delta_y;
            for (int k = 0; k < channel; ++k) {
                boundary_dif_color[i][k] = inner_dif_color[pnt_id[boundary_pnt[i]]][k] =
                        static_cast<double>(getPixel(_dst, tx, ty, k)) -
                        static_cast<double>(getPixel(_src, sx, sy, k));
            }
    }

    // process inner points
    for (auto &x : mesh_pnt) {
        if (boundary_pnt_id.find(x) != boundary_pnt_id.end())
            continue;
        // calculate the coordinate w.r.t. the polygon boundary
        std::vector<double> tangent(bsz);
        for (int k = 0; k < bsz; ++k)
            tangent[k] = getTan(x, boundary_pnt[k], boundary_pnt[(k + 1) % bsz]);
        std::vector<double> weight(bsz);
        double weight_sum = 0;
        for (int k = 0; k < bsz; ++k) {
            weight[k] = (tangent[(k + bsz - 1) % bsz] + tangent[k]) / getDist(x, boundary_pnt[k]);
            weight_sum += weight[k];
        }
        for (int k = 0; k < bsz; ++k) {
            double lambda = weight[k] / weight_sum;
            for (int c = 0; c < channel; ++c)
                inner_dif_color[pnt_id[x]][c] += lambda * boundary_dif_color[k][c];
        }
    }

    // interpolate all the triangles
    for(auto &t: mesh_triangle)
    {
        mesh::Point v1 = t.vertex(0);
        mesh::Point v2 = t.vertex(1);
        mesh::Point v3 = t.vertex(2);
        // bounding box
        double x_min = std::min({v1.x(), v2.x(), v3.x()}) - 1;
        if(x_min < 0) x_min = 0;
        double x_max = std::max({v1.x(), v2.x(), v3.x()}) + 1;
        double y_min = std::min({v1.y(), v2.y(), v3.y()}) - 1;
        if(y_min < 0)y_min = 0;
        double y_max = std::max({v1.y(), v2.y(), v3.y()}) + 1;
        int id[3] = {pnt_id[v1], pnt_id[v2], pnt_id[v3]};
        for(int k = 0; k < channel; ++k)
        {
            auto param = getInterpolationParams(v1, inner_dif_color[id[0]][k], v2, inner_dif_color[id[1]][k], v3, inner_dif_color[id[2]][k]);
            for(int i = x_min;i <= x_max && i < _dst.cols; ++i)
                for(int j = y_min; j <= y_max && j < _dst.rows; ++j)
                {
                    if(inTriangle(v1, v2, v3, mesh::Point(i, j)))
                    {
                        double interpolated = std::get<0>(param) * i + std::get<1>(param) * j + std::get<2>(param) + getPixel(_src, i, j, k);
                        setPixel(_output, static_cast<uchar>(roundAndNormalize(interpolated)), i + delta_x, j + delta_y, k);
                    }
                }
        }
    }
}

// tan(a_i/2) in the paper
double MVCCloner::getTan(const mesh::Point &origin, const mesh::Point &a, const mesh::Point &b) {
    double x0 = a.x() - origin.x(), y0 = a.y() - origin.y(), x1 = b.x() - origin.x(), y1 = b.y() - origin.y();
    double dot_p = x0 * x1 + y0 * y1;
    double sqr_mod = sqrt((x0 * x0 + y0 * y0) * (x1 * x1 + y1 * y1));
    return sqrt((sqr_mod - dot_p) / (sqr_mod + dot_p));
}

double MVCCloner::getDist(const mesh::Point &p1, const mesh::Point &p2) {
    mesh::Point p(p1.x() - p2.x(), p1.y() - p2.y());
    return sqrt(p.x() * p.x() + p.y() * p.y());
}

uchar MVCCloner::getPixel(const cv::Mat &img, int x, int y, int ch) {
    if(x < 0 || x >= img.cols )
	return 0;
    if(y < 0 || y >= img.rows )
	return 0;
    switch (img.type()) {
        case CV_8UC1:
            return img.at<uchar>(y, x);
        case CV_8UC2:
            return img.at<cv::Vec2b>(y, x)[ch];
        case CV_8UC3:
            return img.at<cv::Vec3b>(y, x)[ch];
        case CV_8UC4:
            return img.at<cv::Vec4b>(y, x)[ch];
        default:
            LOG(FATAL) << "Received image format not supported";
    }
}

void MVCCloner::setPixel(cv::Mat &img, uchar val, int x, int y, int ch) {
    if(x < 0 || x >= img.cols )
	return;
    if(y < 0 || y >= img.rows )
	return;
    switch (img.type()) {
        case CV_8UC1:
            img.at<uchar>(y, x) = val;
            break;
        case CV_8UC2:
            img.at<cv::Vec2b>(y, x)[ch] = val;
            break;
        case CV_8UC3:
            img.at<cv::Vec3b>(y, x)[ch] = val;
            break;
        case CV_8UC4:
            img.at<cv::Vec4b>(y, x)[ch] = val;
            break;
        default:
            LOG(FATAL) << "Received image format not supported";
    }
}

std::tuple<double, double, double> MVCCloner::getInterpolationParams(const mesh::Point & p1, double c1, const mesh::Point & p2,
                                                                     double c2, const mesh::Point & p3, double c3) {
    Eigen::Matrix3d mat;
    mat << p1.x(), p1.y(), 1,
    p2.x(), p2.y(), 1,
    p3.x(), p3.y(), 1;
    Eigen::Vector3d color(c1,c2,c3);
    Eigen::Vector3d res = mat.householderQr().solve(color);
    return std::make_tuple(res[0], res[1], res[2]);
}

double MVCCloner::crossProduct(const mesh::Point &p1, const mesh::Point &p2) {
    return p1.x() * p2.y() - p2.x() * p1.y();
}

bool MVCCloner::inTriangle(const mesh::Point &p1, const mesh::Point &p2, const mesh::Point &p3, const mesh::Point &p)
{
    mesh::Point pa(p1.x()- p.x(), p1.y()-p.y());
    mesh::Point pb(p2.x()- p.x(), p2.y()-p.y());
    mesh::Point pc(p3.x()- p.x(), p3.y()-p.y());
    double c1 = crossProduct(pa, pb);
    double c2 = crossProduct(pb, pc);
    double c3 = crossProduct(pc, pa);
    return c1 * c2 >= 0 && c2 * c3 >= 0;
}
