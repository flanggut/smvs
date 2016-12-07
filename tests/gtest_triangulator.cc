/*
 * Copyright (C) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <gtest/gtest.h>

#include "mve/image_io.h"
#include "mve/mesh_io.h"

#include "delaunay_2d.h"
#include "depth_triangulator.h"

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

    EXPECT_EQ(32, faces.size() / 3);
    double const eps = 1e-6;
    for (int i = 0; i < 30; i+= 2)
    {
        EXPECT_NEAR(spiral[i], verts[i / 2 + 4][0], eps);
        EXPECT_NEAR(spiral[i + 1], verts[i / 2 + 4][1], eps);
    }
}

TEST(DepthTriangulator, PixelsForTriangle)
{
    math::Vec3d a(499.0, 499.0, 0.0);
    math::Vec3d b(499.0, 0.0, 0.0);
    math::Vec3d c(0.0, 499.0, 0.0);
    math::Vec3d d(0.0, 0.0, 0.0);
    math::Vec3d e(999.0, 999.0, 0.0);
    math::Vec3d f(999.0, 500.0, 0.0);
    math::Vec3d g(500.0, 999.0, 0.0);
    math::Vec3d h(500.0, 500.0, 0.0);

    std::vector<math::Vec2i> pixels;
    DepthTriangulator::pixels_for_triangle(a, b, c, &pixels);
    DepthTriangulator::pixels_for_triangle(b, c, d, &pixels);
    DepthTriangulator::pixels_for_triangle(e, f, g, &pixels);
    DepthTriangulator::pixels_for_triangle(f, g, h, &pixels);

    EXPECT_EQ(pixels.size(), 1000 * 1000 / 2);

    /* debug write */
//    mve::ByteImage::Ptr image = mve::ByteImage::create(1000, 1000, 1);
//    image->fill(0);
//    for (auto p : pixels)
//    {
//        image->at(p[0], p[1], 0) = 255;
//    }
//    mve::image::save_png_file(image, "/tmp/debug.png");
}

TEST(DepthTriangulator, ApproximateTriangulation)
{
    mve::FloatImage::Ptr dm = mve::FloatImage::create(10, 10, 1);
    mve::CameraInfo cam;
    cam.flen = 1;
    cam.trans[2] = -1.0;
    for (int i = 0; i < dm->width(); ++i)
        for (int j = 0; j < dm->height(); ++j)
            dm->at(i, j, 0) = (MATH_POW2((float)i - 4.5)
                + MATH_POW2((float)j - 4.5)) / 15.0f + 1.0f;

    DepthTriangulator dt(dm, cam);
    mve::TriangleMesh::Ptr approx = dt.approximate_triangulation(27);
    EXPECT_EQ(27, approx->get_vertices().size());
    EXPECT_EQ(38, approx->get_faces().size() / 3);
//    mve::geom::save_mesh(approx, "/tmp/approx.ply");
}
