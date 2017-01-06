/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>

#include "mve/depthmap.h"
#include "mve/mesh_tools.h"
#include "mve/mesh_info.h"
#include "mve/mesh_io.h"

#include "mesh_generator.h"
#include "mesh_simplifier.h"
#include "depth_triangulator.h"

SMVS_NAMESPACE_BEGIN

/* WIP */
void
MeshGenerator::cut_depth_maps(std::vector<mve::FloatImage::Ptr> * depthmaps,
    std::vector<mve::FloatImage::Ptr> * normalmaps)
{
    std::vector<std::future<void>> results;
    std::vector<mve::FloatImage::Ptr> cutmaps(depthmaps->size());
    std::vector<mve::FloatImage::Ptr> cutmaps_j(depthmaps->size());

    for (std::size_t i = 0; i < this->views.size(); ++i)
    results.emplace_back(this->thread_pool.add_task(
        [i, this, &depthmaps, &normalmaps, &cutmaps, &cutmaps_j]
    {
        if (depthmaps->at(i) == nullptr)
            return;
        cutmaps[i] = depthmaps->at(i)->duplicate();

        math::Matrix3f invproj;
        this->views[i]->get_camera().fill_inverse_calibration(
            *invproj, depthmaps->at(i)->width(), depthmaps->at(i)->height());
        mve::image::depthmap_convert_conventions<float>(depthmaps->at(i),
            invproj, false);
        cutmaps_j[i] = cutmaps[i]->duplicate();
    }));
    for(auto && result: results) result.get();
    results.clear();

    std::vector<math::Matrix4f> ctws(this->views.size());
    std::vector<math::Matrix3f> invprojs(this->views.size());
    for (std::size_t i = 0; i < this->views.size(); ++i)
    {
        mve::View::Ptr view = this->views[i];
        view->get_camera().fill_cam_to_world(*ctws[i]);
        view->get_camera().fill_inverse_calibration(*invprojs[i],
            depthmaps->at(i)->width(), depthmaps->at(i)->height());
    }

    for (std::size_t i = 0; i < this->views.size(); ++i)
    results.emplace_back(this->thread_pool.add_task(
        [i, this, &depthmaps, &normalmaps, &cutmaps, &cutmaps_j,
         &ctws, &invprojs]
    {
        mve::View::Ptr view = this->views[i];
        mve::FloatImage::Ptr depth = depthmaps->at(i);
        mve::FloatImage::Ptr normals = normalmaps->at(i);
        mve::FloatImage::Ptr cutmap = cutmaps[i];

        math::Matrix4f const& ctw = ctws[i];
        math::Matrix3f const& invproj = invprojs[i];

        for (int x = 0; x < depth->width(); ++x)
            for (int y = 0; y < depth->height(); ++y)
            {
                float d = cutmap->at(x, y, 0);
                if (d == 0.0)
                    continue;

                /* generate 3D point */
                math::Vec3f pos = mve::geom::pixel_3dpos(x, y, d, invproj);
                pos = ctw.mult(pos, 1.0);
                /* get normal */
                math::Vec3f normal(normals->at(x, y, 0), normals->at(x, y, 1),
                    normals->at(x, y, 2));

                float surface_power = this->view_projs[i].get_surface_power(
                    pos, normal);

                if (surface_power < 0)
                    cutmaps[i]->at(x, y, 0) = 0.0;

                float consistency = 0;

                for (std::size_t j = 0; j < this->views.size(); ++j)
                {
                    if (j == i)
                        continue;
                    mve::FloatImage::ConstPtr normals_j = normalmaps->at(j);
                    math::Matrix4f const& ctw_j = ctws[j];
                    math::Matrix3f const& invproj_j = invprojs[j];

                    math::Vec3f proj = this->view_projs[j].get_proj(pos);
                    if (proj[2] < 0)
                        continue;
                    int xj = static_cast<int>(proj[0] / proj[2]);
                    int yj = static_cast<int>(proj[1] / proj[2]);
                    if (xj < 0 || xj >= depthmaps->at(j)->width()
                        || yj < 0 || yj >= depthmaps->at(j)->height())
                        continue;

                    float dm_j = depthmaps->at(j)->at(xj, yj, 0);
                    if (dm_j == 0.0)
                        continue;

                    float surface_power_j =
                        this->view_projs[j].get_surface_power(pos, normal);

                    /* generate 3D point in neighbor view */
                    math::Vec3f pos_j = mve::geom::pixel_3dpos(xj, yj,
                           cutmaps_j[j]->at(xj, yj, 0), invproj_j);
                    pos_j = ctw_j.mult(pos_j, 1.0);
                    math::Vec3f normal_j(normals_j->at(xj, yj, 0),
                        normals_j->at(xj, yj, 1), normals_j->at(xj, yj, 2));
                    float surface_power_j_j =
                        this->view_projs[j].get_surface_power(pos_j, normal_j);

                    if (dm_j * 1.01 < proj[2])
                        continue;

                    if (dm_j * 0.997 > proj[2])
                    {
                        if (surface_power_j_j > 0.5 * surface_power)
                            consistency -= surface_power_j_j;
                        continue;
                    }
                    if (surface_power_j_j > 2.0 * surface_power
                        || surface_power_j > 2.0 * surface_power)
                    {
                        cutmaps[i]->at(x, y, 0) = 0.0;
                        break;
                    }
                    consistency += surface_power_j_j;
                }
                if (consistency <= 0)
                    cutmaps[i]->at(x, y, 0) = 0.0;
            }
    }));
    /* Wait for finish */
    for(auto && result: results) result.get();

    for (std::size_t i = 0; i < this->views.size(); ++i)
    {
        if (depthmaps->at(i) == nullptr)
            continue;
        depthmaps->at(i) = cutmaps[i]->duplicate();
    }
}

mve::TriangleMesh::Ptr
MeshGenerator::generate_mesh (mve::Scene::ViewList const& inputviews,
    std::string const& image_name, std::string const& dm_name)
{
    /* Look only for valid views with image, depth, and normals */
    std::string nm_name = dm_name + "N";
    std::vector<std::size_t> valid_views;
    for (std::size_t i = 0; i < inputviews.size(); ++i)
    {
        mve::View::Ptr view = inputviews[i];
        if (view == nullptr)
            continue;
        if (!view->has_image(dm_name) || !view->has_image(nm_name) ||
            !view->has_image(image_name))
            continue;

        mve::View::ImageProxy const* proxy = view->get_image_proxy(dm_name);
        this->view_projs.emplace_back(view->get_camera(),
            proxy->width, proxy->height);
        this->views.push_back(view);
    }

    /* Get depth and normals and transform normals to world space */
    std::vector<mve::FloatImage::Ptr> depthmaps(this->views.size());
    std::vector<mve::FloatImage::Ptr> normalmaps(this->views.size());
    std::vector<std::future<void>> load_and_convert;
    for (std::size_t i = 0; i < normalmaps.size(); ++i)
    load_and_convert.emplace_back(this->thread_pool.add_task(
        [i, this, &depthmaps, &normalmaps, &dm_name, &nm_name]
    {
        depthmaps[i] = this->views[i]->get_float_image(dm_name);
        normalmaps[i] = this->views[i]->get_float_image(nm_name);
        mve::FloatImage::Ptr normals = normalmaps[i];
        math::Matrix3f rot;
        this->views[i]->get_camera().fill_cam_to_world_rot(*rot);
        for (int i = 0; i < normals->get_pixel_amount(); ++i)
        {
            math::Vec3f normal(normals->at(i, 0), -normals->at(i, 1),
                -normals->at(i, 2));
            normal = rot * normal;
            normals->at(i, 0) = normal[0];
            normals->at(i, 1) = normal[1];
            normals->at(i, 2) = normal[2];
        }
    }));
    for(auto && result: load_and_convert) result.get();

    /* cut depthmaps */
    if (this->opts.cut_surfaces)
        this->cut_depth_maps(&depthmaps, &normalmaps);

    /* Prepare output mesh. */
    std::vector<std::future<void>> merge_depth;
    mve::TriangleMesh::Ptr pset(mve::TriangleMesh::create());
    std::mutex mesh_mutex;
    for (std::size_t i = 0; i < this->views.size(); ++i)
    merge_depth.emplace_back(this->thread_pool.add_task(
        [this, &mesh_mutex, &image_name, &depthmaps, &normalmaps, i, pset]
    {
        if (this->views[i] == nullptr)
            return;

        if (opts.cut_surfaces)
        {
            this->views[i]->set_image(depthmaps[i], "smvs-cut");
            this->views[i]->save_view();
        }

        mve::TriangleMesh::VertexList& verts(pset->get_vertices());
        mve::TriangleMesh::NormalList& vnorm(pset->get_vertex_normals());
        mve::TriangleMesh::ColorList& vcolor(pset->get_vertex_colors());
        mve::TriangleMesh::ValueList& vvalues(pset->get_vertex_values());
        mve::TriangleMesh::ConfidenceList& vconfs(
            pset->get_vertex_confidences());

        mve::FloatImage::Ptr normals = normalmaps[i];
        mve::TriangleMesh::Ptr m;
        if (this->opts.create_triangle_mesh && this->opts.simplify)
        {
            DepthTriangulator dt(depthmaps[i], this->views[i]->get_camera());
            m = dt.approximate_triangulation(100000);
        }
        else
        {
            m = mve::geom::depthmap_triangulate(depthmaps[i],
            this->views[i]->get_byte_image(image_name),
            this->views[i]->get_camera(), 7.0);
        }

        mve::TriangleMesh::VertexList const& mverts(m->get_vertices());
        mve::TriangleMesh::ColorList const& mvcol(m->get_vertex_colors());
        mve::TriangleMesh::ConfidenceList& mconfs(m->get_vertex_confidences());

        /* Per-vertex confidence down-weighting boundaries. */
        mve::geom::depthmap_mesh_confidences(m, 4);

        std::vector<float> mvscale;
        mvscale.resize(mverts.size(), 0.0f);
        mve::MeshInfo mesh_info(m);
        for (std::size_t j = 0; j < mesh_info.size(); ++j)
        {
            mve::MeshInfo::VertexInfo const& vinf = mesh_info[j];
            for (std::size_t k = 0; k < vinf.verts.size(); ++k)
                mvscale[j] += (mverts[j] - mverts[vinf.verts[k]]).norm();
            mvscale[j] /= static_cast<float>(vinf.verts.size());
            mvscale[j] *= 2.0f;
        }

        mve::TriangleMesh::NormalList smvsnormals;
        smvsnormals.resize(mverts.size());
        for (std::size_t v = 0; v < mverts.size(); ++v)
        {
            math::Vec3f proj = this->view_projs[i].get_proj(mverts[v]);
            int x = static_cast<int>(proj[0] /proj[2]);
            int y = static_cast<int>(proj[1] /proj[2]);
            if (x < 0 || x > normals->width() - 1
                || y < 0 || y > normals->height() - 1)
                    continue;
            smvsnormals[v][0] = normals->at(x, y, 0);
            smvsnormals[v][1] = normals->at(x, y, 1);
            smvsnormals[v][2] = normals->at(x, y, 2);
        }

        std::unique_lock<std::mutex> mesh_lock(mesh_mutex);
        if (this->opts.create_triangle_mesh)
        {
            mve::geom::mesh_merge(m, pset);
        }
        else
        {
            verts.insert(verts.end(), mverts.begin(), mverts.end());
            if (!mvcol.empty())
                vcolor.insert(vcolor.end(), mvcol.begin(), mvcol.end());
            vnorm.insert(vnorm.end(), smvsnormals.begin(), smvsnormals.end());
            vvalues.insert(vvalues.end(), mvscale.begin(), mvscale.end());
            vconfs.insert(vconfs.end(), mconfs.begin(), mconfs.end());
        }
        mesh_lock.unlock();
    }));
    /* Wait for finish */
    for(auto && result: merge_depth) result.get();

    return pset;
}


MeshGenerator::ViewProjection::ViewProjection (mve::CameraInfo const& camera,
    int width, int height)
{
    math::Matrix3f K, R;
    camera.fill_calibration(*K, width, height);
    camera.fill_world_to_cam_rot(*R);

    KR = K * R;
    camera.fill_camera_pos(*t);
    t = KR * t;
}

math::Vec3f
MeshGenerator::ViewProjection::ViewProjection::get_proj (math::Vec3f pos) const
{
    float u = KR.row(0).dot(pos) - t[0];
    float v = KR.row(1).dot(pos) - t[1];
    float w = KR.row(2).dot(pos) - t[2];
    return math::Vec3f(u, v, w);
}

float
MeshGenerator::ViewProjection::ViewProjection::get_surface_power (
    math::Vec3f const& pos, math::Vec3f const& normal)
{
    math::Vec3f u_dx, v_dx;
    float u = KR.row(0).dot(pos) - t[0];
    float v = KR.row(1).dot(pos) - t[1];
    float w = KR.row(2).dot(pos) - t[2];

    float denom = w * w;

    u_dx[0] = (KR(0, 0) * w - KR(2, 0) * u) / denom;
    u_dx[1] = (KR(0, 1) * w - KR(2, 1) * u) / denom;
    u_dx[2] = (KR(0, 2) * w - KR(2, 2) * u) / denom;

    v_dx[0] = (KR(1, 0) * w - KR(2, 0) * v) / denom;
    v_dx[1] = (KR(1, 1) * w - KR(2, 1) * v) / denom;
    v_dx[2] = (KR(1, 2) * w - KR(2, 2) * v) / denom;

    float power = -normal.dot(u_dx.cross(v_dx));
    return power;
}

SMVS_NAMESPACE_END
