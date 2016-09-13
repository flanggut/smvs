/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_MESH_SIMPLIFY_HEADER
#define SMVS_MESH_SIMPLIFY_HEADER

#include <queue>

#include "mve/mesh.h"
#include "mve/mesh_info.h"
#include "math/matrix.h"

#include "defines.h"

SMVS_NAMESPACE_BEGIN

/* 
 * Simplifies a triangle mesh to a given percent of vertices using
 * Garlands quadric metric
 */
class MeshSimplifier
{
public:
    MeshSimplifier (mve::TriangleMesh::ConstPtr mesh);

    /* Simplify the mesh to given percentage (e.g. 30% means 70% are removed) */
    mve::TriangleMesh::Ptr get_simplified (float percent);

private:
    struct SimplifyEdge
    {
        double cost;
        std::size_t v1;
        std::size_t v2;
        math::Vec3d new_vert;
        math::Matrix4d quadric;

        bool operator > (SimplifyEdge const& rhs) const
        {
            return this->cost > rhs.cost;
        }
    };

private:
    void compute_initial_quadrics (void);
    SimplifyEdge create_simplify_edge (std::size_t v1, std::size_t v2);
    void fill_queue (void);

private:
    mve::TriangleMesh::ConstPtr input_mesh;
    mve::TriangleMesh::Ptr mesh;
    mve::MeshInfo mesh_info;
    std::vector<math::Matrix4d> quadrics;

    std::priority_queue<SimplifyEdge, std::vector<SimplifyEdge>,
        std::greater<SimplifyEdge>> removal_queue;
};

inline
MeshSimplifier::MeshSimplifier (mve::TriangleMesh::ConstPtr mesh)
    : input_mesh(mesh)
    , mesh_info(mesh)
{
}

SMVS_NAMESPACE_END

#endif /* SMVS_MESH_SIMPLIFY_HEADER */
