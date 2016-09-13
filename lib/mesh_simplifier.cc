/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <set>
#include <iostream>

#include "mve/mesh_tools.h"
#include "math/matrix_tools.h"

#include "mesh_simplifier.h"

SMVS_NAMESPACE_BEGIN

/* Edge collapse from MVE repo (c) Simon Fuhrmann 
   Updated by Fabian Langguth to include non manifold check and fix bugs */
/* TODO: make edge collapse accessible in MVE rather than having it here */
bool
edge_collapse (mve::TriangleMesh::Ptr mesh, mve::MeshInfo& mesh_info,
    std::size_t v1, std::size_t v2, math::Vec3f const& new_vert,
    std::vector<std::size_t> const& afaces,
    float acos_threshold = 0.95f)
{
    mve::TriangleMesh::FaceList& faces = mesh->get_faces();
    mve::TriangleMesh::VertexList& verts = mesh->get_vertices();

    /* Test if the hypothetical vertex destroys geometry. */
    mve::MeshInfo::VertexInfo& vinfo1 = mesh_info[v1];
    for (std::size_t i = 0; i < vinfo1.verts.size(); ++i)
    {
        std::size_t ip1 = (i + 1) % vinfo1.verts.size();
        if (vinfo1.verts[i] == v2 || vinfo1.verts[ip1] == v2)
            continue;

        math::Vec3f const& av1 = verts[vinfo1.verts[i]];
        math::Vec3f const& av2 = verts[vinfo1.verts[ip1]];
        math::Vec3f n1 = (av1 - verts[v1]).cross(av2 - verts[v1]).normalized();
        math::Vec3f n2 = (av1 - new_vert).cross(av2 - new_vert).normalized();

        float dot = n1.dot(n2);
        if (MATH_ISNAN(dot)|| dot < acos_threshold)
            return false;
    }

    mve::MeshInfo::VertexInfo& vinfo2 = mesh_info[v2];
    for (std::size_t i = 0; i < vinfo2.verts.size(); ++i)
    {
        std::size_t ip1 = (i + 1) % vinfo2.verts.size();
        if (vinfo2.verts[i] == v1 || vinfo2.verts[ip1] == v1)
            continue;
        math::Vec3f const& av1 = verts[vinfo2.verts[i]];
        math::Vec3f const& av2 = verts[vinfo2.verts[ip1]];
        math::Vec3f n1 = (av1 - verts[v2]).cross(av2 - verts[v2]).normalized();
        math::Vec3f n2 = (av1 - new_vert).cross(av2 - new_vert).normalized();

        float dot = n1.dot(n2);
        if (MATH_ISNAN(dot) || dot < acos_threshold)
            return false;
    }

    /* Test if collapse creates nonmanifold */
    std::size_t joint_neighbors = 0;
    for (std::size_t i = 0; i < vinfo1.verts.size(); ++i)
        for (std::size_t j = 0; j < vinfo2.verts.size(); ++j)
            if (vinfo1.verts[i] == vinfo2.verts[j])
            {
                joint_neighbors += 1;
                continue;
            }
    if (joint_neighbors != 2)
        return false;

    /* Test succeeded. Assign new vertex position to v1. */
    verts[v1] = new_vert;

    /* Delete the two faces adjacent to the collapsed edge. */
    std::size_t v3 = 0, v4 = 0;
    for (std::size_t i = 0; i < 3; ++i)
    {
        std::size_t fid1 = afaces[0] * 3 + i;
        std::size_t fid2 = afaces[1] * 3 + i;
        if (faces[fid1] != v1 && faces[fid1] != v2)
            v3 = faces[fid1];
        if (faces[fid2] != v1 && faces[fid2] != v2)
            v4 = faces[fid2];
        faces[fid1] = 0;
        faces[fid2] = 0;
    }

    /* Update vertex info for vertices adjcent to v2, replacing v2 with v1. */
    for (std::size_t i = 0; i < vinfo2.verts.size(); ++i)
    {
        std::size_t const vert_id = vinfo2.verts[i];
        if (vert_id != v1 && vert_id != v3 && vert_id != v4)
            mesh_info[vert_id].replace_adjacent_vertex(v2, v1);
    }

    /* Update faces adjacent to v2 replacing v2 with v1. */
    for (std::size_t i = 0; i < vinfo2.faces.size(); ++i)
        for (std::size_t j = 0; j < 3; ++j)
            if (faces[vinfo2.faces[i] * 3 + j] == v2)
                faces[vinfo2.faces[i] * 3 + j] = static_cast<unsigned int>(v1);

    /* Update vertex info for v3 and v4: remove v2, remove deleted faces. */
    mve::MeshInfo::VertexInfo& vinfo3 = mesh_info[v3];
    vinfo3.remove_adjacent_face(afaces[0]);
    vinfo3.remove_adjacent_vertex(v2);
    mve::MeshInfo::VertexInfo& vinfo4 = mesh_info[v4];
    vinfo4.remove_adjacent_face(afaces[1]);
    vinfo4.remove_adjacent_vertex(v2);

    /* Update vinfo for v1: Remove v2, remove collapsed faces, add v2 faces. */
    vinfo1.remove_adjacent_vertex(v2);
    vinfo1.remove_adjacent_face(afaces[0]);
    vinfo1.remove_adjacent_face(afaces[1]);
    for (std::size_t i = 0; i < vinfo2.faces.size(); ++i)
    if (vinfo2.faces[i] != afaces[0] && vinfo2.faces[i] != afaces[1])
            vinfo1.faces.push_back(vinfo2.faces[i]);

    /* Clear verts for v1 and re-add with new faces */
    vinfo1.verts.clear();
    mesh_info.update_vertex(*mesh, v1);

    /* Update vertex info for v2. */
    vinfo2.faces.clear();
    vinfo2.verts.clear();
    vinfo2.vclass = mve::MeshInfo::VERTEX_CLASS_UNREF;

    return true;
}

void
MeshSimplifier::compute_initial_quadrics (void)
{
    mve::TriangleMesh::VertexList & verts = this->mesh->get_vertices();
    mve::TriangleMesh::FaceList & faces = this->mesh->get_faces();

    this->quadrics.resize(this->mesh_info.size());
    for (std::size_t v = 0; v < this->mesh_info.size(); ++v)
    {
        this->quadrics[v].fill(0);
        mve::MeshInfo::VertexInfo const& vinfo = this->mesh_info.at(v);
        for (std::size_t f = 0; f < vinfo.faces.size(); ++f)
        {
            math::Vec3d faceverts[3];
            for (std::size_t i = 0; i < 3; ++i)
                faceverts[i] = verts[faces[vinfo.faces[f] * 3 + i]];
            math::Vec3d normal = (faceverts[1] - faceverts[0]).cross(
                faceverts[2] - faceverts[0]).normalized();

            math::Vec4d face_plane(normal, -normal.dot(faceverts[0]));
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    this->quadrics[v](r, c) += face_plane[r] * face_plane[c];
        }
    }
}

MeshSimplifier::SimplifyEdge
MeshSimplifier::create_simplify_edge (std::size_t v1, std::size_t v2)
{
    mve::TriangleMesh::VertexList & verts = this->mesh->get_vertices();
    SimplifyEdge sedge;
    sedge.v1 = v1;
    sedge.v2 = v2;
    sedge.quadric = this->quadrics[v1] + this->quadrics[v2];

    /* Compute optimal vertex position */
    math::Matrix4d Q = sedge.quadric;
    Q(3,0) = Q(3,1) = Q(3,2) = 0;
    Q(3,3) = 1.0;
    double det = math::matrix_determinant(Q);
    if (det != 0.0)
    {
        /* compute optimal vertex position */
        Q = math::matrix_inverse(Q);
        math::Vec4d rhs(0.0, 0.0, 0.0, 1.0);
        math::Vec4d vert4 = Q*rhs;
        sedge.new_vert = math::Vec3d(*vert4);
        sedge.cost = std::max(0.0, (sedge.quadric * vert4).dot(vert4));
    }
    else
    {
        /* Choose average between v1 and v2 */
        sedge.new_vert = (verts[v1] + verts[v2]) * 0.5;
        math::Vec4d vert4(sedge.new_vert, 1.0);
        sedge.cost = std::max(0.0, (sedge.quadric * vert4).dot(vert4));
    }

    return sedge;
}

void
MeshSimplifier::fill_queue (void)
{
    /* find all edges to simplify */
    std::set<std::pair<std::size_t, std::size_t>> edges;
    for (std::size_t v = 0; v < this->mesh_info.size(); ++v)
    {
        mve::MeshInfo::VertexInfo const& vinfo = this->mesh_info.at(v);
        if (vinfo.vclass != mve::MeshInfo::VERTEX_CLASS_SIMPLE)
            continue;
        for (std::size_t n = 0; n < vinfo.verts.size(); ++n)
            if (vinfo.verts[n] < v)
                edges.emplace(vinfo.verts[n], v);
            else
                edges.emplace(v, vinfo.verts[n]);
    }
    /* Fill initial queue for edge collapses */
    for (auto const& edge : edges)
        this->removal_queue.emplace(
            this->create_simplify_edge(edge.first, edge.second));
}

mve::TriangleMesh::Ptr
MeshSimplifier::get_simplified (float percent)
{
    this->mesh = this->input_mesh->duplicate();
    this->compute_initial_quadrics();

    mve::TriangleMesh::VertexList & verts = this->mesh->get_vertices();

    this->fill_queue();

    float initial_size = verts.size();
    std::size_t target_remove = initial_size - (initial_size * percent / 100);
    std::size_t removed = 0;
    while (!removal_queue.empty() && removed < target_remove)
    {
        SimplifyEdge const& sedge = removal_queue.top();
        std::size_t const v1 = sedge.v1;
        std::size_t const v2 = sedge.v2;

        /* Skip already collapsed vertices */
        if (this->mesh_info[v1].vclass == mve::MeshInfo::VERTEX_CLASS_UNREF
            || this->mesh_info[v2].vclass == mve::MeshInfo::VERTEX_CLASS_UNREF)
        {
            removal_queue.pop();
            continue;
        }

        std::vector<std::size_t> afaces;
        this->mesh_info.get_faces_for_edge(v1, v2, &afaces);

        /* Collapse edge */
        bool collapsed = edge_collapse(this->mesh, this->mesh_info,
            v1, v2, sedge.new_vert, afaces);

        /* Cannot collapse, remove edge from queue */
        if (!collapsed)
        {
            removal_queue.pop();
            continue;
        }

        /* Collapse sucessfull */
        removed += 1;

        /* Update Quartic for new vertex */
        this->quadrics[v1] = sedge.quadric;

        /* add new edges to queue */
        mve::MeshInfo::VertexInfo const& vinfo = this->mesh_info.at(v1);
        for (std::size_t n = 0; n < vinfo.verts.size(); ++n)
            this->removal_queue.emplace(
                this->create_simplify_edge(v1, vinfo.verts[n]));

        removal_queue.pop();
    }
    /* Cleanup invalid triangles and unreferenced vertices. */
    mve::geom::mesh_delete_unreferenced(mesh);
    return this->mesh;
}

SMVS_NAMESPACE_END
