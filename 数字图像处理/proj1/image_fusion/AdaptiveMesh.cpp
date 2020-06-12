/*
 * Created by zx on 19-3-8.
 */

#include "AdaptiveMesh.h"
#include <glog/logging.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace mesh {

    AdaptiveMesh::AdaptiveMesh(std::vector<Point> &polygon) : _polygon(polygon) {
        _mesh_ptr = new ConstrainedDelaunayTriangulation();
        _mesh_ptr->insert(polygon.begin(), polygon.end());
        // add constraints
        for (auto it = polygon.begin(); it != polygon.end(); ++it) {
            auto next = std::next(it);
            if (next == polygon.end())
                _mesh_ptr->insert_constraint(*it, *(polygon.begin()));
            else
                _mesh_ptr->insert_constraint(*it, *next);
        }
        _mesher_ptr = new Mesher(*_mesh_ptr);
        _mesher_ptr->set_criteria(Criteria(0.125, 0));
        _mesher_ptr->refine_mesh();
        LOG(INFO) << "Generated mesh with vertices: " << numVertices() << " and faces: " << numFaces();
    }

    AdaptiveMesh::~AdaptiveMesh() {
        if (_mesher_ptr)
            delete _mesher_ptr;
        if (_mesh_ptr)
            delete _mesh_ptr;
    }

    void AdaptiveMesh::calcPoints() {
        _result_points.clear();
        _point_id.clear();
        _triangle_inside.clear();
        int id = 0;
        for (auto it = _mesh_ptr->finite_faces_begin(); it != _mesh_ptr->finite_faces_end(); ++it) {
            Triangle triangle = _mesh_ptr->triangle(it);
            Point v[3] = {triangle.vertex(0), triangle.vertex(1), triangle.vertex(2)};
            if (it->is_in_domain()) // skip triangles which are not in domain
                _triangle_inside.emplace_back(triangle);
            for (int j = 0; j < 3; ++j)
                if (_point_id.find(v[j]) == _point_id.end())
                {
                    _point_id[v[j]] = id++; _result_points.push_back(v[j]);
                }

        }
        LOG(INFO) << "Inner triangles: " << _triangle_inside.size() << "\n";
    }

    cv::Mat AdaptiveMesh::visualize(const cv::Size &size) const {
        cv::Mat img(size, CV_8UC1, cv::Scalar(0));
        for(const auto & t: _triangle_inside)
        {
            auto v1 = t.vertex(0);
            auto v2 = t.vertex(1);
            auto v3 = t.vertex(2);
            cv::line(img, cv::Point(v1.x(), v1.y()), cv::Point(v2.x(), v2.y()), cv::Scalar(255));
            cv::line(img, cv::Point(v3.x(), v3.y()), cv::Point(v2.x(), v2.y()), cv::Scalar(255));
            cv::line(img, cv::Point(v1.x(), v1.y()), cv::Point(v3.x(), v3.y()), cv::Scalar(255));
        }
        return img;
    }
}