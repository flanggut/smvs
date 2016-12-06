/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>

#include "delaunay_2d.h"

SMVS_NAMESPACE_BEGIN

/* ----------------------- Geometric Tools ------------------------ */
namespace
{
    double triangle_area(math::Vec2d const& a, math::Vec2d const& b,
        math::Vec2d const& c)
    {
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    }

    bool in_circle(math::Vec2d const& a, math::Vec2d const& b,
        math::Vec2d const& c, math::Vec2d const& p)
    {
        return (a.square_norm() * triangle_area(b, c, p) -
            b.square_norm() * triangle_area(a, c, p) +
            c.square_norm() * triangle_area(a, b, p) -
            p.square_norm() * triangle_area(a, b, c)) > 0;
    }

    int ccw(math::Vec2d const& a, math::Vec2d const& b, math::Vec2d const& c)
    {
        return (triangle_area(a, b, c) > 0);
    }

    int right_of(math::Vec2d const& p, math::Vec2d const& orig,
        math::Vec2d const& dest)
    {
        return ccw(p, dest, orig);
    }

    bool on_edge(math::Vec2d const& p, math::Vec2d const& orig,
        math::Vec2d const& dest)
    {
        double const eps = 1e-5;
        double t1 = (p - orig).norm();
        double t2 = (p - dest).norm();
        if (t1 < eps || t2 < eps)
            return true;
        double t3 = (orig - dest).norm();
        if (t1 > t3 || t2 > t3)
            return false;
        /* point to line distance */
        double dist = ((dest[1] - orig[1]) * p[0])
            - ((dest[0] - orig[0]) * p[1])
            + dest[0] * orig[1] - dest[1] * orig[0];
        dist /= t3;
        return (std::fabs(dist) < eps);
    }
}

/* ---------------------- Edge Manipulation ----------------------- */
void
Delaunay2D::debug_print_edge(Edge::Ptr e)
{
    std::cout << this->edge_orig(e) << ", " << this->edge_dest(e) << "; "
        << e->left() << " " << e->right() << std::endl;
}

void
Delaunay2D::flip_edge (Edge::Ptr e)
{
    Edge::Ptr prev = e->o_prev();
    Edge::Ptr inv_prev = e->inv()->o_prev();

    /* remove edge and inv from their vertices */
    Edge::splice(e, prev);
    Edge::splice(e->inv(), inv_prev);
    /* add edge and inv to their next vertices ccw */
    Edge::splice(e, prev->l_next());
    Edge::splice(e->inv(), inv_prev->l_next());
    /* change vertex data */
    e->set_vertex_ids(prev->dest(), inv_prev->dest());

    /* change face data */
    /* left face */
    e->l_next()->set_left_face(e->left());
    e->l_prev()->set_left_face(e->left());
    /* right face */
    e->d_next()->set_left_face(e->right());
    e->o_prev()->set_left_face(e->right());

    this->triangles[e->left()].start = e;
    this->triangles[e->right()].start = e->inv();
    this->recently_changed.insert(e->left());
    this->recently_changed.insert(e->right());
}

Edge::Ptr
Delaunay2D::connect_edges (Edge::Ptr a, Edge::Ptr b)
{
    Edge::Ptr e = QuadEdge::create_edge(&this->q_edges);
    Edge::splice(e, a->l_next());
    Edge::splice(e->inv(), b);
    e->set_vertex_ids(a->dest(), b->orig());
    return e;
}

void
Delaunay2D::delete_edge (Edge::Ptr e)
{
    Edge::splice(e, e->o_prev());
    Edge::splice(e->inv(), e->inv()->o_prev());
}

/* -------------------- Triangle Manipulation --------------------- */
math::Vec3ui
Delaunay2D::Triangle::get_vertices (void) const
{
    math::Vec3ui verts;
    Edge::Ptr edge = this->start;
    for (int v = 0; v < 3; ++v)
    {
        verts[v] = static_cast<unsigned int>(edge->orig());
        edge = edge->l_next();
    }
    return verts;
}

void
Delaunay2D::fill_triangle_vertices (std::size_t triangle,
    double * vertices) const
{
    math::Vec3ui vertex_ids = this->triangles[triangle].get_vertices();
    for (int i = 0; i < 9; ++i)
        vertices[i] = this->vertices[vertex_ids[i / 3]][i % 3];
}

/* ------------------------ Delaunay Main ------------------------- */
Delaunay2D::Delaunay2D(math::Vec2d min, math::Vec2d max, double z)

{
    math::Vec3d bot_left(min[0], min[1], z);
    math::Vec3d bot_right(max[0], min[1], z);
    math::Vec3d top_left(min[0], max[1], z);
    math::Vec3d top_right(max[0], max[1], z);
    this->initialize(bot_left, bot_right, top_left, top_right);
}

Delaunay2D::Delaunay2D(math::Vec3d p1, math::Vec3d p2, math::Vec3d p3,
    math::Vec3d p4)

{
    this->initialize(p1, p2, p3, p4);
}

void
Delaunay2D::initialize(math::Vec3d p1, math::Vec3d p2, math::Vec3d p3,
    math::Vec3d p4)
{
    this->vertices.push_back(p1);
    this->vertices.push_back(p2);
    this->vertices.push_back(p3);
    this->vertices.push_back(p4);

    Edge::Ptr e1 = QuadEdge::create_edge(&this->q_edges);
    e1->set_vertex_ids(0, 1);
    Edge::Ptr e2 = QuadEdge::create_edge(&this->q_edges);
    Edge::splice(e1->inv(), e2);
    e2->set_vertex_ids(1, 2);
    Edge::Ptr e3 = QuadEdge::create_edge(&this->q_edges);
    Edge::splice(e2->inv(), e3);
    e3->set_vertex_ids(2, 0);
    Edge::splice(e3->inv(), e1);
    this->triangles.emplace_back(e1);
    e1->set_left_face(0);
    e2->set_left_face(0);
    e3->set_left_face(0);


    Edge::Ptr e4 = QuadEdge::create_edge(&this->q_edges);
    Edge::splice(e1->inv(), e4);
    e4->set_vertex_ids(1, 3);
    Edge::Ptr e5 = QuadEdge::create_edge(&this->q_edges);
    Edge::splice(e4->inv(), e5);
    e5->set_vertex_ids(3, 2);
    Edge::splice(e5->inv(), e2->inv());
    this->triangles.emplace_back(e4);
    e4->set_left_face(1);
    e5->set_left_face(1);
    e5->l_next()->set_left_face(1);
    e5->l_next()->inv()->set_right_face(1);

    this->start = e1;
}

Edge::Ptr
Delaunay2D::locate(math::Vec2d const& p, Edge::Ptr start_edge)
{
    Edge::Ptr e = start_edge;
    while (true)
    {
        if ((p[0] == this->vertices[e->orig()][0]
            && p[1] == this->vertices[e->orig()][1])
            || (p[0] == this->vertices[e->dest()][0]
            && p[1] == this->vertices[e->dest()][1]))
            return e;
        else if (right_of(p, this->edge_orig(e), this->edge_dest(e)))
            e = e->inv();
        else if (!right_of(p, this->edge_orig(e->o_next()),
            this->edge_dest(e->o_next())))
            e = e->o_next();
        else if (!right_of(p, this->edge_orig(e->d_prev()),
           this->edge_dest(e->d_prev())))
            e = e->d_prev();
        else
            return e;
    }
}

void
Delaunay2D::insert_point(math::Vec3d const& p3d, std::size_t triangle)
{
    this->recently_changed.clear();
    math::Vec2d p(p3d[0], p3d[1]);
    Edge::Ptr e;
    if (triangle == std::size_t(-1))
        e = this->locate(p, this->start);
    else
        e = this->locate(p, this->triangles[triangle].start);

    if ((p == this->edge_orig(e)) || (p == edge_dest(e)))
        return;

    if (on_edge(p, this->edge_orig(e), this->edge_dest(e)))
    {
        e = e->o_prev();
        this->delete_edge(e->o_next());
    }

    Edge::Ptr base = QuadEdge::create_edge(&this->q_edges);
    this->vertices.push_back(p3d);
    base->set_vertex_ids(e->orig(), this->vertices.size() - 1);
    base->set_right_face(e->left());
    this->triangles[e->left()].start = e;
    this->recently_changed.insert(e->left());

    Edge::splice(base, e);
    Edge::Ptr start = base;
    for (int i = 0; i < 2; ++i)
    {
        base = connect_edges(e, base->inv());
        base->set_left_face(e->left());
        e = base->o_prev();
        this->triangles.emplace_back(e);
        this->recently_changed.insert(this->triangles.size() - 1);
        e->set_left_face(this->triangles.size() - 1);
        base->set_right_face(e->left());
    }
    if (e->l_next() != start)
    {
        base = connect_edges(e, base->inv());
        base->set_left_face(e->left());
        e = base->o_prev();
        this->triangles[e->left()].start = e;
        this->recently_changed.insert(e->left());
        base->set_right_face(e->left());
    }
    start->set_left_face(e->left());

    while (true)
    {
        Edge::Ptr t = e->o_prev();
        if (right_of(this->edge_dest(t), this->edge_orig(e),
                this->edge_dest(e)) &&
            in_circle(this->edge_orig(e), this->edge_dest(t),
                this->edge_dest(e), p))
        {
            flip_edge(e);
            e = e->o_prev();
        }
        else if (e->o_next() == start)
            return;
        else
            e = e->o_next()->l_prev();
    }

}

mve::TriangleMesh::Ptr
Delaunay2D::get_mesh (void) const
{
    mve::TriangleMesh::Ptr mesh = mve::TriangleMesh::create();

    mve::TriangleMesh::VertexList & verts = mesh->get_vertices();
    verts.clear();
    verts.reserve(this->vertices.size());
    for (auto v : this->vertices)
        verts.push_back(v);

    mve::TriangleMesh::FaceList & faces = mesh->get_faces();
    faces.reserve(this->triangles.size() * 3);
    for (auto & T : this->triangles)
    {
        math::Vec3ui verts = T.get_vertices();
        faces.push_back(verts[0]);
        faces.push_back(verts[1]);
        faces.push_back(verts[2]);
    }

    return mesh;
}

SMVS_NAMESPACE_END
