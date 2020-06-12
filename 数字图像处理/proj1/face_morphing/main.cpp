#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <glog/logging.h>
#include <cstdio>
#include <Eigen/Dense>
#include <vector>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Delaunay;
typedef K::Point_2 Point;

struct PairHash {
    template<typename T1, typename T2>
    size_t operator()(std::pair<T1, T2> const &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

typedef std::unordered_map<std::pair<int, int>, int, PairHash> PointHashMap;

Delaunay triangulate(cv::Mat img, const std::vector<cv::Point> &points) {
    std::vector<Point> points_copy;
    for (const auto &p : points) {
        points_copy.emplace_back(Point(p.x, p.y));
    }
    Delaunay dt;
    dt.insert(points_copy.begin(), points_copy.end());
    return dt;
}

std::pair<cv::Mat, std::vector<cv::Point>> load_image(const std::string &fname) {
    cv::Mat img = cv::imread(fname);
    auto annot_name = fname + ".txt";
    auto fp = std::fopen(annot_name.c_str(), "r");
    if (fp == nullptr) LOG(FATAL) << "Failed to load annotation file: " << annot_name;
    fscanf(fp, "%*d %*d %*d %*d");
    int x, y;
    std::vector<cv::Point> points;
    while (~fscanf(fp, "%d %d", &x, &y)) {
        points.emplace_back(cv::Point(y, x));
    }
    auto w = img.cols, h = img.rows;
    points.emplace_back(cv::Point(0, 0));
    points.emplace_back(cv::Point(w / 2, 0));
    points.emplace_back(cv::Point(w - 1, 0));
    points.emplace_back(cv::Point(w - 1, h / 2));
    points.emplace_back(cv::Point(w - 1, h - 1));
    points.emplace_back(cv::Point(w / 2, h - 1));
    points.emplace_back(cv::Point(0, h - 1));
    points.emplace_back(cv::Point(0, h / 2));
    return std::make_pair(img, points);
}

inline float cross_product(const cv::Point &p1, const cv::Point &p2) {
    return p1.x * p2.y - p2.x * p1.y;
}

inline bool in_triangle(const cv::Point &p0, const cv::Point &p1, const cv::Point &p2, const cv::Point &p) {
    cv::Point pa = p0 - p, pb = p1 - p, pc = p2 - p;
    auto c1 = cross_product(pa, pb), c2 = cross_product(pb, pc), c3 = cross_product(pc, pa);
    return c1 * c2 >= 0 && c2 * c3 >= 0;
}

inline cv::Point transform(const cv::Point &p, const std::pair<Eigen::Vector3f, Eigen::Vector3f> trans) {
    Eigen::Vector3f p_;
    p_ << p.x, p.y, 1;
    float x = trans.first.dot(p_);
    float y = trans.second.dot(p_);
    return cv::Point(int(std::round(x)), int(std::round(y)));
}

inline int get_intermediate(int x, int y, float ratio) {
    return int(std::round(x * (1 - ratio) + y * ratio));
}

inline uint8_t clip(float x) {
    int xx = (int) round(x);
    return xx <= 0 ? 0 : (xx >= 255 ? 255 : xx);
}

inline cv::Vec3b get_intermediate(const cv::Vec3b &v1, const cv::Vec3b &v2, float ratio) {
    return cv::Vec3b(clip(v1[0] * (1 - ratio) + v2[0] * ratio), clip(v1[1] * (1 - ratio) + v2[1] * ratio),
                     clip(v1[2] * (1 - ratio) + v2[2] * ratio));
}

cv::Vec3b get_pixel(cv::Mat img, cv::Point pt) {
    return img.at<cv::Vec3b>(pt);
}

std::pair<Eigen::Vector3f, Eigen::Vector3f>
solve_trans(const cv::Point &s0, const cv::Point &s1, const cv::Point &s2, const cv::Point &d0, const cv::Point &d1,
            const cv::Point &d2) {
    Eigen::Matrix3f A;
    Eigen::Vector3f b1, b2;
    A << s0.x, s0.y, 1, s1.x, s1.y, 1, s2.x, s2.y, 1;
    b1 << d0.x, d1.x, d2.x;
    b2 << d0.y, d1.y, d2.y;
    auto qr = A.colPivHouseholderQr();
    Eigen::Vector3f x1 = qr.solve(b1);
    Eigen::Vector3f x2 = qr.solve(b2);
    return {x1, x2};
}

void transform_and_fill(cv::Mat src, cv::Mat dst, cv::Mat out, cv::Point s0, cv::Point s1, cv::Point s2, cv::Point d0,
                        cv::Point d1, cv::Point d2, float ratio) {
    cv::Point m0 = cv::Point(get_intermediate(s0.x, d0.x, ratio), get_intermediate(s0.y, d0.y, ratio));
    cv::Point m1 = cv::Point(get_intermediate(s1.x, d1.x, ratio), get_intermediate(s1.y, d1.y, ratio));
    cv::Point m2 = cv::Point(get_intermediate(s2.x, d2.x, ratio), get_intermediate(s2.y, d2.y, ratio));
    // solve transformation M --> S, M --> D
    auto T1 = solve_trans(m0, m1, m2, s0, s1, s2);
    auto T2 = solve_trans(m0, m1, m2, d0, d1, d2);
    // interpolate and fill color
    auto x_min = std::min(m0.x, std::min(m1.x, m2.x));
    auto x_max = std::max(m0.x, std::max(m1.x, m2.x));
    auto y_min = std::min(m0.y, std::min(m1.y, m2.y));
    auto y_max = std::max(m0.y, std::max(m1.y, m2.y));
    for (int x = x_min; x <= x_max; ++x) {
        for (int y = y_min; y <= y_max; ++y) {
            cv::Point p(x, y);
            if (in_triangle(m0, m1, m2, p)) {
                auto p_src = transform(p, T1);
                auto p_dst = transform(p, T2);
                auto cc_src = get_pixel(src, p_src);
                auto cc_dst = get_pixel(dst, p_dst);
                out.at<cv::Vec3b>(p) = get_intermediate(cc_src, cc_dst, ratio);
            }
        }
    }
}

template<typename T>
inline void draw_triangle(cv::Mat img, T f, const cv::Scalar &color = cv::Scalar(0, 0, 255)) {
    auto _p0 = f->vertex(0)->point();
    auto _p1 = f->vertex(1)->point();
    auto _p2 = f->vertex(2)->point();
    auto p0 = cv::Point(_p0.x(), _p0.y());
    auto p1 = cv::Point(_p1.x(), _p1.y());
    auto p2 = cv::Point(_p2.x(), _p2.y());
    cv::line(img, p0, p1, color);
    cv::line(img, p1, p2, color);
    cv::line(img, p2, p0, color);
}

inline void draw_triangle(cv::Mat img, const cv::Point &p0, const cv::Point &p1, const cv::Point &p2,
                          const cv::Scalar &color = cv::Scalar(0, 0, 255)) {
    cv::line(img, p0, p1, color);
    cv::line(img, p1, p2, color);
    cv::line(img, p2, p0, color);
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    google::LogToStderr();
    if (argc != 5) {
        LOG(FATAL) << "Usage: " << argv[0] << " [src_img] [target_img] [output_img] [ratio (0-100)]";
    }
    auto src = load_image(argv[1]);
    auto dst = load_image(argv[2]);
    auto out_fname = argv[3];
    int ratio_ = atoi(argv[4]);
    if (ratio_ < 0 || ratio_ > 100) LOG(FATAL) << "Invalid ratio: " << ratio_;
    if (ratio_ == 0) cv::imwrite(out_fname, src.first);
    if (ratio_ == 100) cv::imwrite(out_fname, dst.first);
    float ratio = ratio_ / 100.f;
    auto out = cv::Mat(int(std::round((1 - ratio) * src.first.rows + ratio * dst.first.rows)),
                       int(std::round((1 - ratio) * src.first.cols + ratio * dst.first.cols)), CV_8UC3,
                       cv::Scalar(0, 0, 0));
    if (src.second.size() != dst.second.size()) LOG(FATAL) << "Check failed: source and dst annotation size not equals";
    // construct point to index map (for proper ordering)
    PointHashMap idx_map;
    for (int i = 0; i < src.second.size(); ++i) {
        cv::Point p1 = src.second[i], p2 = dst.second[i];
        std::pair<int,int> pair = {p1.x, p1.y};
        if(idx_map.find(pair) != idx_map.end())
            LOG(FATAL) << "Encountered duplicate entry: " << pair.first << ", " << pair.second;
        idx_map[pair] = i;
    }
    auto tri = triangulate(src.first, src.second);
    Delaunay::Finite_faces_iterator fit;
    cv::Mat tmp = src.first.clone();
    cv::Mat tmp_2 = dst.first.clone();

    for (auto face = tri.finite_faces_begin();
         face != tri.finite_faces_end(); ++face) {
        draw_triangle(tmp, face);
        cv::imshow("Delaunay_src", tmp);
        auto _v0 = face->vertex(0)->point(), _v1 = face->vertex(1)->point(), _v2 = face->vertex(2)->point();
        int idx0 = idx_map[{_v0.x(), _v0.y()}], idx1 = idx_map[{_v1.x(), _v1.y()}], idx2 = idx_map[{_v2.x(), _v2.y()}];
        assert(idx0 >= 0 && idx0 < src.second.size() && idx1 >= 0 && idx1 < src.second.size() && idx2 >= 0 &&
               idx2 < src.second.size());
        auto s0 = cv::Point(_v0.x(), _v0.y()), s1 = cv::Point(_v1.x(), _v1.y()), s2 = cv::Point(_v2.x(), _v2.y());
        auto d0 = cv::Point(dst.second[idx0].x, dst.second[idx0].y),
                d1 = cv::Point(dst.second[idx1].x, dst.second[idx1].y),
                d2 = cv::Point(dst.second[idx2].x, dst.second[idx2].y);
        draw_triangle(tmp_2, d0, d1, d2);
        cv::imshow("Delaunay_dst", tmp_2);
        transform_and_fill(src.first, dst.first, out, s0, s1, s2, d0, d1, d2, ratio);
        cv::imshow("Result", out);
        cv::waitKey(1);
    }
    cv::imwrite(out_fname, out);
    cv::waitKey();
    return 0;
}