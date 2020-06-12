/*
 * Created by zx on 19-3-8.
 */

#ifndef MVC_ADAPTIVEMESH_H
#define MVC_ADAPTIVEMESH_H

#include <vector>
#include <map>
#include <memory>
#include <opencv2/core/types.hpp>
#include <CGAL/Cartesian.h>
#include <CGAL/Triangulation_vertex_base_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>

namespace mesh {
    typedef CGAL::Exact_predicates_inexact_constructions_kernel _Kernel;
    typedef CGAL::Triangulation_vertex_base_2<_Kernel> _Vertexb;
    typedef CGAL::Delaunay_mesh_face_base_2<_Kernel> _Faceb;
    typedef CGAL::Triangulation_data_structure_2<_Vertexb, _Faceb> _Tds;
    typedef CGAL::Constrained_Delaunay_triangulation_2<_Kernel, _Tds> ConstrainedDelaunayTriangulation;
    typedef CGAL::Delaunay_mesh_size_criteria_2<ConstrainedDelaunayTriangulation> Criteria;
    typedef CGAL::Delaunay_mesher_2<ConstrainedDelaunayTriangulation, Criteria> Mesher;
    typedef ConstrainedDelaunayTriangulation::Triangle Triangle;
    typedef ConstrainedDelaunayTriangulation::Point Point;

    class AdaptiveMesh {
    public:
        explicit AdaptiveMesh(std::vector<Point> &polygon);

        int numVertices() const { return _mesh_ptr->number_of_vertices(); };

        int numFaces() const { return _mesh_ptr->number_of_faces(); };

        void calcPoints();   // must be called before accessing results

        std::vector<Point> getPoints() const { return _result_points; }

        std::vector<Triangle> getTriangles() const { return _triangle_inside; }

        std::vector<Point> getBoundaryPoints() const {return _polygon; }

        std::map<Point, int> getPointId() const { return _point_id; }

        cv::Mat visualize(const cv::Size &size) const;

        ~AdaptiveMesh();

    private:
        std::vector<Point> &_polygon;
        ConstrainedDelaunayTriangulation *_mesh_ptr;
        Mesher *_mesher_ptr;
        std::vector<Point> _result_points;
        std::map<Point, int> _point_id;
        std::vector<Triangle> _triangle_inside;
    };
}


#endif //MVC_ADAPTIVEMESH_H
