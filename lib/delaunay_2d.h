/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_DELAUNAY_2D_HEADER
#define SMVS_DELAUNAY_2D_HEADER

#include <memory>

#include "math/vector.h"
#include "mve/mesh.h"

#include "defines.h"
#include "quad_edge.h"

SMVS_NAMESPACE_BEGIN

/*
 * Incremental 2D Delaunay Triangulation based on:
 *   Leonidas J. Guibas and Jorge Stolfi,
 *   Primitives for the Manipulation of General Subdivisions and the
 *   Computation of Voronoi Diagrams, ACM ToG 4(2):74–123, April 1985
 */

class Delaunay2D
{
public:
    /// Initialize triangulation to rectangular domain
    Delaunay2D (math::Vec2d min, math::Vec2d max, double z);

    void insert_point (math::Vec3d const& p3d);

    mve::TriangleMesh::Ptr get_mesh (void);

private:
    void flip_edge (Edge::Ptr e);
    Edge::Ptr connect_edges (Edge::Ptr a, Edge::Ptr b);
    void delete_edge (Edge::Ptr e);
    Edge::Ptr locate (math::Vec2d const& p);
    math::Vec2d edge_orig (Edge::Ptr e);
    math::Vec2d edge_dest (Edge::Ptr e);

    void debug_print_edge (Edge::Ptr e);

private:
    struct Triangle
    {
        Triangle (Edge::Ptr start) : start(start) { }
        Edge::Ptr start;
        math::Vec3ui get_vertices (void);
    };

    Edge::Ptr start;
    std::vector<math::Vec3d> vertices;
    std::vector<Triangle> triangles;
    std::vector<std::unique_ptr<QuadEdge>> q_edges;
};

/* ------------------------ Implementation ------------------------ */

inline math::Vec2d
Delaunay2D::edge_orig (Edge::Ptr e)
{
    return math::Vec2d(*this->vertices[e->orig()]);
}

inline math::Vec2d
Delaunay2D::edge_dest (Edge::Ptr e)
{
    return math::Vec2d(*this->vertices[e->dest()]);
}


SMVS_NAMESPACE_END

#endif /* SMVS_DELAUNAY_2D_HEADER */
