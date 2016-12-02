/*
 * Copyright (C) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <gtest/gtest.h>

#include "delaunay_2d.h"
#include "mve/mesh_io.h"

using namespace smvs;

void
debug_print_mesh (mve::TriangleMesh::Ptr mesh)
{
    std::cout << "Vertices: " << std::endl;
    for (auto v : mesh->get_vertices())
    {
        std::cout << v[0] << " " << v[1] << " " << v[2] << std::endl;
    }
    std::cout << "Faces: " << std::endl;
    int v = 0;
    for (std::size_t f = 0; f < mesh->get_faces().size(); ++f)
    {
        std::cout << mesh->get_faces()[f];
        if (++v < 3)
        {
            std::cout << " ";
        }
        else
        {
            v = 0;
            std::cout << std::endl;
        }
    }
}

TEST(Delaunay2DTest, InitialMesh)
{
    math::Vec2d min(0.0, 0.0);
    math::Vec2d max(1.0, 1.0);

    Delaunay2D delaunay(min, max, 0.0);

    mve::TriangleMesh::Ptr mesh = delaunay.get_mesh();
}

TEST(Delaunay2DTest, InsertPoint)
{
    math::Vec2d min(-4.0, -4.0);
    math::Vec2d max(4.0, 4.0);

    double spiral[30]
    {
        0.0, 0.0, -0.416, 0.909, -1.35, 0.436,
        -1.64, -0.549, -1.31, -1.51, -0.532, -2.17,
        0.454, -2.41, 1.45, -2.21, 2.29, -1.66,
        2.88, -0.838, 3.16, 0.131, 3.12, 1.14,
        2.77, 2.08, 2.16, 2.89, 1.36, 3.49
    };

    Delaunay2D delaunay(min, max, 0.0);
    for (int i = 0; i < 30; i+= 2)
        delaunay.insert_point(math::Vec3d(spiral[i], spiral[i+1], 0.0));

    mve::TriangleMesh::Ptr mesh = delaunay.get_mesh();
    mve::TriangleMesh::VertexList const& verts = mesh->get_vertices();
    mve::TriangleMesh::FaceList const& faces = mesh->get_faces();

    EXPECT_EQ(99, faces.size());
    double const eps = 1e-6;
    for (int i = 0; i < 30; i+= 2)
    {
        EXPECT_NEAR(spiral[i], verts[i / 2 + 4][0], eps);
        EXPECT_NEAR(spiral[i + 1], verts[i / 2 + 4][1], eps);
    }
}
