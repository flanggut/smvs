/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_MESH_GENERATOR_HEADER
#define SMVS_MESH_GENERATOR_HEADER

#include "mve/scene.h"

#include "defines.h"
#include "thread_pool.h"

SMVS_NAMESPACE_BEGIN

class MeshGenerator
{
public:
    struct Options
    {
        std::size_t num_threads;
        bool cut_surfaces = false;
        bool create_triangle_mesh = false;
        bool simplify = false;

        Options (void)
        {
            num_threads = std::thread::hardware_concurrency();
        }
    };

public:
    MeshGenerator (Options const& opts);
    
    mve::TriangleMesh::Ptr generate_mesh (mve::Scene::ViewList const& views,
        std::string const& image_name, std::string const& dm_name);

private:
    void cut_depth_maps (std::vector<mve::FloatImage::Ptr> * depthmaps,
        std::vector<mve::FloatImage::Ptr> * normalmaps);

private:
    struct ViewProjection
    {
        ViewProjection (mve::CameraInfo const& camera, int width, int height);

        math::Vec3f get_proj(math::Vec3f pos) const;
        float get_surface_power(math::Vec3f const& pos,
            math::Vec3f const& normal);

        math::Matrix3f KR;
        math::Vec3f t;
    };

    Options const& opts;

    ThreadPool thread_pool;
    std::vector<mve::View::Ptr> views;
    std::vector<ViewProjection> view_projs;
};

inline
MeshGenerator::MeshGenerator (MeshGenerator::Options const& opts)
    : opts(opts)
    , thread_pool(opts.num_threads)
{
}

SMVS_NAMESPACE_END

#endif /* SMVS_MESH_GENERATOR_HEADER */
