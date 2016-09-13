/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <smmintrin.h> // SSE4_1

#include "util/timer.h"
#include "mve/image_io.h"
#include "mve/image_tools.h"
#include "math/matrix_svd.h"

#include "depth_optimizer.h"
#include "correspondence.h"
#include "surface_derivative.h"
#include "light_optimizer.h"
#include "conjugate_gradient.h"
#include "sgm_stereo.h"

// minimal value for reweighting linear system to optimize L1 norm
#define R_FACTOR 1e-4

SMVS_NAMESPACE_BEGIN

DepthOptimizer::DepthOptimizer (StereoView::Ptr main_view,
    std::vector<StereoView::Ptr> const& sub_views,
    mve::Bundle::ConstPtr bundle, Options const& opts)
    : opts(opts), bundle(bundle), main_view(main_view), sub_views(sub_views)
    , lighting(nullptr)
{
    this->prepare_correspondences();
}

void
DepthOptimizer::create_initial_surface (void)
{
    if (this->opts.use_sgm)
    {
        mve::FloatImage::Ptr init = this->main_view->get_sgm_depth();
        this->surface = Surface::create(bundle, main_view, 4, init);
        this->sgm_depth = init;
    }
    else
        this->surface = Surface::create(bundle, main_view, 5);
}

void
DepthOptimizer::optimize (void)
{
    this->create_initial_surface();

    if (this->opts.debug_lvl > 0)
        std::cout << "########### Scale "
            << this->surface->get_scale() << " ###########" << std::endl;

    util::WallTimer scale_timer;
    this->main_view->set_scale(this->surface->get_scale(),
        this->opts.debug_lvl > 2);
    for (auto view : this->sub_views)
        view->set_scale(this->surface->get_scale(), this->opts.debug_lvl > 2);

    if (this->opts.debug_lvl > 1)
        this->main_view->write_depth_to_view(this->surface->get_depth_map(),
            "smvs-initial");

    if (this->opts.debug_lvl > 2)
    {
        /* Test initial reprojections */
        this->create_subview_surfaces();
        for (std::size_t i = 0; i < this->sub_views.size(); ++i)
            this->reproject_neighbor(i);
    }

    /* run optimization on coarsest level */
    this->run_newton_iterations(this->opts.num_iterations);

    /* debug output */
    if (this->opts.debug_lvl > 0)
        std::cout << "Scale " << this->surface->get_scale()
        << " took " << scale_timer.get_elapsed_sec() << "s" << std::endl;
    this->write_debug_depth();

    while (this->surface->get_scale() > this->opts.min_scale
        && this->surface->get_scale() > 0)
    {
        if (this->opts.debug_lvl > 0)
            std::cout << "########### Scale "
                << this->surface->get_scale() - 1 << " ###########" << std::endl;
        scale_timer.reset();

        /* subdivide surface to new scale */
        this->surface->subdivide_patches();
        this->main_view->set_scale(this->surface->get_scale(),
            this->opts.debug_lvl > 2);
        for (auto view : this->sub_views)
            view->set_scale(this->surface->get_scale(),
                this->opts.debug_lvl > 2);
        this->write_debug_depth();

        /* fill non-existing patches from initialization */
        this->surface->fill_patches_from_depth();

        /* use lighting */
        if (this->opts.use_lighting && this->surface->get_scale() < 4)
        {
            if (this->opts.debug_lvl > 0)
                std::cout << "######## with Lighting ########" << std::endl;
            LightOptimizer light_opt(this->surface, this->main_view);
            this->lighting = light_opt.fit_lighting_to_image(
               this->main_view->get_shading_image());
        }
        
        if (this->opts.debug_lvl > 1 && this->lighting != nullptr)
        {
            mve::FloatImage::Ptr shaded = this->lighting->render_normal_map(
                this->get_normals());
            this->main_view->write_image_to_view(shaded, "lighting-shaded");
            mve::FloatImage::Ptr sphere =
                this->lighting->get_rendered_sphere(555);
            this->main_view->write_image_to_view(sphere, "lighting-sphere");
        }

        /* run optimization */
        this->run_newton_iterations(this->opts.num_iterations);

        if (this->opts.debug_lvl > 0)
            std::cout << "Scale " << this->surface->get_scale()
                << " took " << scale_timer.get_elapsed_sec() << "s" << std::endl;
        this->write_debug_depth();
    }

    /* Write final output */
    if (this->opts.debug_lvl > 1 && this->lighting != nullptr)
    {
        mve::FloatImage::Ptr shaded = this->lighting->render_normal_map(
            this->get_normals());
        this->main_view->write_image_to_view(shaded, "lighting-shaded");
        mve::FloatImage::Ptr sphere = this->lighting->get_rendered_sphere(555);
        this->main_view->write_image_to_view(sphere, "lighting-sphere");
        mve::FloatImage::Ptr albedo =
            this->main_view->get_linear_image()->duplicate();
        for (int p = 0; p < albedo->get_pixel_amount(); ++p)
            if (shaded->at(p) > 0.0)
               for(int c = 0; c < 3; ++c)
                  albedo->at(p, c) /= shaded->at(p);
            else
               for(int c = 0; c < 3; ++c)
                   albedo->at(p, c) = 0.0;
        this->main_view->write_image_to_view(albedo, "implicit-albedo");
    }

    this->main_view->write_depth_to_view(this->surface->get_depth_map(),
        this->opts.output_name);
    this->main_view->write_image_to_view(this->get_normals(),
        this->opts.output_name + "N");
}

void
DepthOptimizer::run_newton_iterations (int num_iters)
{
    DenseVector gradient;
    DenseVector prev_gradient;
    SparseMatrix hessian;
    SparseMatrix precond;
    std::vector<double> delta;
    std::vector<double> depth_updates;
    std::vector<math::Vec2d> projections1;
    std::vector<math::Vec2d> projections2;

    this->main_gradients = this->main_view->get_image_gradients();
    this->main_gradients_linear = this->main_view->get_linear_gradients();

    bool finished = false;
    for (int iter = 0; iter < num_iters; ++iter)
    {
        std::size_t num_valid = 0;
        for (auto p : this->surface->get_patches())
            if (p != nullptr)
                num_valid++;
        if (this->opts.debug_lvl > 0 && iter == 0)
            std::cout << "Surface Status - "
                "Valid patches: " << num_valid << std::endl;

        if (iter == 0)
        {
            this->create_subview_surfaces();
            int deleted = std::numeric_limits<int>::max();
            while (deleted > 10)
                deleted = this->cut_boundaries();
        }

        double update = 100.0;
        unsigned int newton_step = 0;
        unsigned int linear_iterations = 0;
        float timer_build_step = 0.0;
        float timer_solve_step = 0.0;
        for (; newton_step < 200 && update > 0.01; ++newton_step)
        {
            prev_gradient = gradient;

            util::WallTimer newton_timer;
            /* construct newton step */
            this->construct_newton_step(&hessian, &gradient, &precond);
            timer_build_step += newton_timer.get_elapsed();

            if (this->opts.debug_lvl > 2)
                std::cout << "Building g+H entries took "
                    << newton_timer.get_elapsed() << " ms" << std::endl;

            if (newton_step == 0)
                prev_gradient = gradient;

            this->write_debug_depth();

            /* solve newton step */
            ConjugateGradient::Options cg_opts;
            cg_opts.max_iterations = 200;
            cg_opts.error_tolerance = gradient.norm() * 0.01;
            ConjugateGradient cg_solver(cg_opts);
            ConjugateGradient::Status cg_status;
            DenseVector x;
            gradient.negate_self();

            newton_timer.reset();
            cg_status = cg_solver.solve(hessian, gradient, &x, &precond);

            timer_solve_step += newton_timer.get_elapsed();
            linear_iterations += cg_status.num_iterations;
            if (this->opts.debug_lvl > 2)
                std::cout << "Solving linear system took: "
                    << newton_timer.get_elapsed() << " ms"
                    << " Num iterations " << cg_status.num_iterations
                    <<  std::endl;

            /* update variables */
            delta.resize(x.size());
            std::copy(x.begin(), x.end(), delta.begin());
            if (std::isnan(delta[0]))
                break;

            /* update nodes and compute reprojection difference */
            this->fill_node_reprojections(&projections1);
            this->surface->update_nodes(delta, &depth_updates);
            this->fill_node_reprojections(&projections2);

            double sum_diff = 0;
            for (std::size_t p = 0; p < projections1.size(); ++p)
                sum_diff += (projections1[p] - projections2[p]).norm();
            update = sum_diff / (double) projections1.size();
            if (this->opts.debug_lvl > 2)
                std::cout << "Avg delta: " << update << std::endl;
        }

        if (this->opts.debug_lvl > 0)
        {
            std::cout << "### Iteration: " << iter << std::endl;
            std::cout << "Number of Newton steps: " << newton_step << std::endl;
            std::cout << "Avg construction time: "
                << timer_build_step / newton_step << "ms" << std::endl;
            std::cout << "Avg solver time: "
                << timer_solve_step / newton_step << "ms" << std::endl;
            std::cout << "Avg solver iterations: "
                << linear_iterations / newton_step << std::endl;
        }
        this->write_debug_depth();

        // FIXME: Check non converged nodes
        std::size_t counter = 0;
        for (std::size_t i = 0; i < depth_updates.size(); ++i)
            if (depth_updates[i] > 0.005)
            {
                this->surface->delete_node(i);
                counter++;
            }
        this->surface->remove_patches_without_nodes();

        /* expand surface and cut occlusion boundaries */
        if(!finished)
        {
            int deleted = std::numeric_limits<int>::max();
            while (deleted > 10)
                deleted = this->cut_boundaries();
            if (!this->opts.use_sgm)
            {
                this->surface->expand();
                this->write_debug_depth("-exp");
                this->create_subview_surfaces();
                deleted = std::numeric_limits<int>::max();
                while (deleted > 10)
                    deleted = this->cut_boundaries();
            }
        }
        this->surface->remove_isolated_patches();
        if (finished)
            break;

        std::size_t num_valid_new = 0;
        for (auto p : this->surface->get_patches())
            if (p != nullptr)
                num_valid_new++;

        /* check surface changes for convergence */
        double change = 1.0 - (double)std::min(num_valid_new, num_valid)
            / (double)std::max(num_valid_new, num_valid);

        if (this->opts.debug_lvl > 0)
            std::cout << "Surface change: " << change << " Valid patches: "
                << num_valid << " -> " << num_valid_new << std::endl;
        this->write_debug_depth();

        if (iter > 0 && (num_valid_new <= num_valid ||
            change < 0.05 * this->surface->get_scale()))
            finished = true;
    }
}

int
DepthOptimizer::cut_boundaries (void)
{
    int deleted = 0;
    Surface::PatchList const& patches = this->surface->get_patches();

    /* first remove depth discontinuities */
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id)
    {
        if (patches[patch_id] == nullptr)
            continue;

        patches[patch_id]->fill_values_at_nodes(&pixels, &depths,
            &depth_derivatives);

        math::Matrix3f invproj;
        this->main_view->get_camera().fill_inverse_calibration(*invproj,
            this->main_view->get_width(), this->main_view->get_height());

        std::multimap<double, std::size_t> depth_p;
        for (std::size_t i = 0; i < 4; ++i)
            depth_p.insert(std::make_pair(depths[i], i));

        double dd_factor = 7.0;
        auto iter_first = depth_p.begin();
        auto iter_last = std::prev(depth_p.end());

        if (iter_first->second + iter_last->second == 3)
            dd_factor *= MATH_SQRT2;
        double threshold = dd_factor * iter_first->first * invproj[0]
            * this->surface->get_patchsize();

        double dist = iter_last->first - iter_first->first;
        if (dist > threshold)
        {
            this->surface->delete_patch(patch_id);
            deleted += 1;
        }
    }

    /* remove high error patches at the border of the surface */
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id)
    {
        if (patches[patch_id] == nullptr)
            continue;

        std::size_t node_ids[4];
        this->surface->fill_node_ids_for_patch(patch_id, node_ids);

        double error = this->mse_for_patch(patch_id);
        double tex_score = this->tex_score_for_patch(patch_id);
        for (std::size_t node = 0; node < 4; ++node)
        {
            Surface::NodeList node_neighbors;
            this->surface->fill_node_neighbors(node_ids[node], &node_neighbors);
            int num_invalid = 0;
            for (auto neighbor : node_neighbors)
                if (neighbor == nullptr)
                    num_invalid += 1;
            if (num_invalid > 1)
                if (error > 0.05
                    || (this->surface->get_scale() > 1 && tex_score < 0.005))
                {
                    this->surface->delete_patch(patch_id);
                    deleted += 1;
                    break;
                }
        }
    }
    this->surface->remove_nodes_without_patch();
    return deleted;
}

void
DepthOptimizer::create_subview_surfaces (void)
{
    Surface::PatchList const & patches = this->surface->get_patches();
    this->subsurfaces.clear();
    this->subsurfaces.resize(patches.size());

    int num_total = 0;
    std::vector<mve::FloatImage::Ptr> depth_caches;
    depth_caches.resize(this->sub_views.size());

    for (std::size_t sub_id = 0; sub_id < this->sub_views.size(); ++sub_id)
    {
        int const sub_width = this->sub_views[sub_id]->get_width();
        int const sub_height = this->sub_views[sub_id]->get_height();

        depth_caches[sub_id] = mve::FloatImage::create(
            sub_width + 1, sub_height + 1, 1);
        depth_caches[sub_id]->fill(10000.0);
    }

    /* first pass: find minimal depth */
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id)
    {
        if (patches[patch_id] == nullptr)
            continue;
        num_total += 1;
        std::vector<math::Vec2d> coords;
        std::vector<double> depths;
        patches[patch_id]->fill_values_at_pixels(&coords, &depths);

        for (std::size_t sub_id = 0; sub_id < this->sub_views.size(); ++sub_id)
        {
            double const sub_width = static_cast<double>(
                this->sub_views[sub_id]->get_width());
            double const sub_height = static_cast<double>(
                this->sub_views[sub_id]->get_height());

            for (std::size_t i = 0; i < coords.size(); i++)
            {
                Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                    coords[i][0] + 0.5, coords[i][1] + 0.5, depths[i]);
                math::Vec2d proj;
                C.fill(*proj);
                proj[0] -= 0.5;
                proj[1] -= 0.5;
                double const cutoffset = 10.0;
                if (proj[0] < cutoffset || proj[0] >= sub_width - cutoffset ||
                    proj[1] < cutoffset || proj[1] >= sub_height - cutoffset)
                    break;

                int cx = static_cast<int>(proj[0]);
                int cy = static_cast<int>(proj[1]);
                for (int x = -1; x < 2; ++x)
                    for (int y = -1; y < 2; ++y)
                if (C.get_depth() < depth_caches[sub_id]->at(cx+x, cy+y, 0))
                    depth_caches[sub_id]->at(cx+x, cy+y, 0) = C.get_depth();
            }
        }
    }

    /* second pass: keep ids near minimal depth */
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id)
    {
        if (patches[patch_id] == nullptr)
            continue;

        std::vector<math::Vec2d> coords;
        std::vector<double> depths;
        patches[patch_id]->fill_values_at_pixels(&coords, &depths);

        if (this->opts.use_sgm)
        {
            bool has_sgm_depth = false;
            for (std::size_t i = 0; i < coords.size(); i++)
                if (this->sgm_depth->at(coords[i][0], coords[i][1], 0) != 0.0)
                    has_sgm_depth = true;
            if (!has_sgm_depth)
                continue;
        }

        std::vector<double> sub_nccs;
        for (std::size_t sub_id = 0; sub_id < this->sub_views.size(); ++sub_id)
            sub_nccs.emplace_back(this->ncc_for_patch(patch_id, sub_id));

        for (std::size_t sub_id = 0; sub_id < this->sub_views.size(); ++sub_id)
        {
            double const sub_width = static_cast<double>(
                this->sub_views[sub_id]->get_width());
            double const sub_height = static_cast<double>(
                this->sub_views[sub_id]->get_height());

            bool success = true;
            for (std::size_t i = 0; i < coords.size() && success; i++)
            {
                Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                    coords[i][0] + 0.5, coords[i][1] + 0.5, depths[i]);
                math::Vec2d proj;
                C.fill(*proj);
                proj[0] -= 0.5;
                proj[1] -= 0.5;
                double const cutoffset = 10;
                if (proj[0] < cutoffset || proj[0] >= sub_width - cutoffset ||
                    proj[1] < cutoffset || proj[1] >= sub_height - cutoffset)
                {
                    success = false;
                    break;
                }

                int cx = static_cast<int>(proj[0]);
                int cy = static_cast<int>(proj[1]);
                for (int x = -1; x < 2; ++x)
                    for (int y = -1; y < 2; ++y)
                    if (C.get_depth() * 0.97 >
                        depth_caches[sub_id]->at(cx + x, cy + y, 0))
                        success = false;
            }
            if (!success)
                continue;

            std::vector<math::Vec2d> node_coords;
            std::vector<double> node_depths;
            std::vector<math::Vec2d> node_depths_derivs;
            patches[patch_id]->fill_values_at_pixels(&coords, &depths);
            patches[patch_id]->fill_values_at_nodes(&node_coords, &node_depths,
                &node_depths_derivs);

            if (!this->opts.use_sgm)
            {
                double max = 0.0;
                for (int i = 0; i < 4; ++i)
                {
                    Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                         node_coords[i][0] + 0.5, node_coords[i][1] + 0.5,
                         node_depths[i],
                         node_depths_derivs[i][0], node_depths_derivs[i][1]);
                    math::Matrix2d jac;
                    C.fill_jacobian(*jac);
                    double S[2];
                    math::matrix_svd<double>(*jac, 2, 2, nullptr, S, nullptr);
                    double sigma0 = MATH_POW2(std::max(S[0], S[1]));
                    double sigma1 = MATH_POW2(std::min(S[0], S[1]));
                    max = std::max(max, sigma0 / sigma1);
                }
                if (max > 4.0)
                    continue;
            }

            /* filter possible occlusions through unreconstructed geometry */
            if (!opts.use_sgm && sub_nccs[sub_id] < 0.0)
                continue;

            /* Patch is visible in neigbor view, add to sub_ids */
            this->subsurfaces[patch_id].push_back(sub_id);
        }
    }

    std::size_t num_invalid_patches = 0;
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id)
    {
        Surface::Patch::Ptr patch = patches[patch_id];
        if (patch == nullptr)
            continue;
        if (this->subsurfaces[patch_id].size() < 1)
        {
            this->surface->delete_patch(patch_id);
            num_invalid_patches += 1;
        }
    }
    if (num_invalid_patches > 0)
        this->surface->remove_nodes_without_patch();
    if (this->opts.debug_lvl > 0)
        std::cout << "Removed " << num_invalid_patches << " patches "
            "due to occlusions." << std::endl;
}

void
DepthOptimizer::get_non_converged_nodes(std::vector<math::Vec2d> const& proj1,
    std::vector<math::Vec2d> const& proj2, std::vector<std::size_t> * node_ids)
{
    node_ids->clear();
    Surface::NodeList const& nodes = this->surface->get_nodes();
    Surface::PatchList const& patches = this->surface->get_patches();
    std::vector<double> diffs(nodes.size(), 0.0);

    std::size_t proj_id = 0;
    for (int patch_id = 0; patch_id < (int)patches.size(); ++patch_id)
    {
        SurfacePatch::Ptr patch = patches[patch_id];
        if (patch == nullptr)
            continue;

        std::size_t node_ids[4];
        this->surface->fill_node_ids_for_patch(patch_id, node_ids);
        for (std::size_t i = 0; i < 4; ++i)
        {
            std::size_t node_id = node_ids[i];
            Surface::Node::Ptr node = nodes[node_id];
            for (std::size_t j = 0; j < this->subsurfaces[patch_id].size(); ++j)
            {
                diffs[node_id] += (proj1[proj_id] - proj2[proj_id]).norm()
                    / this->subsurfaces[patch_id].size();
                proj_id += 1;
            }
        }
    }
    if (proj_id != proj1.size())
    {
        std::cout << " Warning: possible error while computing"
            "non-converged nodes" << std::endl;
    }
    for (std::size_t i = 0; i < diffs.size(); ++i)
        if (diffs[i] > 1.5)
            node_ids->push_back(i);
}


void
DepthOptimizer::fill_node_reprojections(std::vector<math::Vec2d> * proj)
{
    proj->clear();
    Surface::NodeList const& nodes = this->surface->get_nodes();
    Surface::PatchList const& patches = this->surface->get_patches();
    std::vector<math::Vec2d> node_coords;
    this->surface->fill_node_coords(&node_coords);

    for (int patch_id = 0; patch_id < (int)patches.size(); ++patch_id)
    {
        SurfacePatch::Ptr patch = patches[patch_id];
        if (patch == nullptr)
            continue;

        std::size_t node_ids[4];
        this->surface->fill_node_ids_for_patch(patch_id, node_ids);
        for (std::size_t i = 0; i < 4; ++i)
        {
            std::size_t node_id = node_ids[i];
            Surface::Node::Ptr node = nodes[node_id];
            for (std::size_t j = 0; j < this->subsurfaces[patch_id].size(); ++j)
            {
                std::size_t sub_id = this->subsurfaces[patch_id][j];
                Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                     node_coords[node_id][0], node_coords[node_id][1], node->f);
                math::Vec2d projection;
                C.fill(*projection);
                proj->push_back(projection);
            }
        }
    }
}

void
DepthOptimizer::prepare_correspondences (void)
{
    this->Mi.resize(this->sub_views.size());
    this->ti.resize(this->sub_views.size());
    mve::CameraInfo main_cam = this->main_view->get_camera();
    for (std::size_t i = 0; i < this->sub_views.size(); ++i)
    {
        mve::CameraInfo  sub_cam = this->sub_views[i]->get_camera();
        math::Matrix3f M;
        math::Vec3f t;
        main_cam.fill_reprojection(sub_cam, this->main_view->get_width(),
            this->main_view->get_height(), this->sub_views[i]->get_width(),
            this->sub_views[i]->get_height(), *M, *t);
            
        for (int j = 0; j < 9; ++j)
            this->Mi[i][j] = M[j];
        for (int j = 0; j < 3; ++j)
            this->ti[i][j] = t[j];
    }
}

void
DepthOptimizer::construct_newton_step (SparseMatrix * hessian,
    DenseVector * gradient, SparseMatrix * precond)
{
    Surface::PatchList const & patches = this->surface->get_patches();
    std::size_t const num_params = this->surface->get_num_nodes() * 4;
    int const pixels_per_patch = MATH_POW2(this->surface->get_patchsize());
    std::vector<double> node_derivatives;
    node_derivatives.resize(pixels_per_patch * 96);
    for (int i = 0; i < pixels_per_patch; ++i)
        this->surface->fill_node_derivatives_for_pixel(i,
            &node_derivatives[i * 96],
            &node_derivatives[i * 96 + 24],
            &node_derivatives[i * 96 + 48],
            &node_derivatives[i * 96 + 72]);

    gradient->resize(num_params);
    gradient->fill(0);

    std::map<std::size_t, SparseMatrix::Block> hessian_blocks;

    this->j_grad_subs.resize(this->sub_views.size());
    this->jac_entries.resize(this->sub_views.size() * 16);

    double sub_gradient[16];
    double sub_hessian[256];

    for (int patch_id = 0; patch_id < (int)patches.size(); ++patch_id)
    {
        SurfacePatch::Ptr patch = patches[patch_id];
        if (patch == nullptr)
            continue;
        std::size_t node_ids[4];
        this->surface->fill_node_ids_for_patch(patch_id, node_ids);

        std::fill(sub_gradient, sub_gradient + 16, 0.0);
        std::fill(sub_hessian, sub_hessian + 256, 0.0);

        this->jacobian_entries_for_patch(patch_id, node_derivatives,
            sub_gradient, sub_hessian);

        /* compute gradient block */
        for (int node = 0; node < 4; ++node)
            for (int value = 0; value < 4; ++value)
                gradient->at(node_ids[node] * 4 + value) +=
                    sub_gradient[node * 4 + value];

        /* compute hessian block */
        for (std::size_t node1 = 0; node1 < 16; ++node1)
            for (std::size_t node2 = node1; node2 < 16; ++node2)
            {
                std::size_t const block_id1 = node_ids[node1 / 4] *
                    num_params + node_ids[node2 / 4];
                std::size_t const block_id2 = node_ids[node1 / 4] +
                    num_params * node_ids[node2 / 4];
                std::size_t const block_offset_x = node1 % 4;
                std::size_t const block_offset_y = node2 % 4;
                hessian_blocks[block_id1].values
                    [block_offset_x + 4 * block_offset_y]
                    += sub_hessian[node1 * 16 + node2];
                if(node1 != node2)
                    hessian_blocks[block_id2].values
                        [block_offset_x * 4 + block_offset_y]
                        += sub_hessian[node1 * 16 + node2];
            }
    }
    hessian->allocate(num_params, num_params);
    precond->allocate(num_params, num_params);
    
    SparseMatrix::Blocks hessian_block_vec;
    hessian_block_vec.reserve(this->surface->get_num_nodes() * 9);
    SparseMatrix::Blocks precond_block_vec;
    precond_block_vec.reserve(this->surface->get_num_nodes());
    
    for (auto & e : hessian_blocks)
    {
        e.second.row = 4 * e.first % num_params;
        e.second.col = 4 * e.first / num_params;
        hessian_block_vec.push_back(e.second);
        if(e.second.row == e.second.col)
            precond_block_vec.push_back(e.second);
    }

    hessian->set_from_blocks(hessian_block_vec);
    precond->set_from_blocks(precond_block_vec);
    precond->invert_blocks_inplace();
}

void
DepthOptimizer::jacobian_entries_for_patch(std::size_t patch_id,
    std::vector<double> const& node_derivatives, double * gradient,
    double * hessian_entries)
{
    Surface::Patch::Ptr patch = this->surface->get_patches()[patch_id];
    std::size_t num_sds = this->subsurfaces[patch_id].size() + 1;
    num_sds = num_sds * (num_sds + 1);
    p_weights.resize(num_sds);
    p_diffs.resize(num_sds);

    int sampling = 4;
    if (this->surface->get_scale() < 5)
        sampling = 2;
    if (this->surface->get_scale() < 3)
        sampling = 1;

    patch->fill_values_at_pixels(&pixels, &depths, &depth_derivatives,
        &depth_2nd_derivatives, &pids, sampling);

    for (std::size_t i = 0; i < pixels.size(); ++i)
    {
        this->grad_main[0] = this->main_gradients->at(
            pixels[i][0], pixels[i][1], 0);
        this->grad_main[1] = this->main_gradients->at(
            pixels[i][0], pixels[i][1], 1);

        double const* dn00 = &node_derivatives[pids[i] * 96];

        for (std::size_t j = 0; j < this->subsurfaces[patch_id].size(); ++j)
        {
            std::size_t sub_id = this->subsurfaces[patch_id][j];
            mve::FloatImage::ConstPtr sub_gradients =
                this->sub_views[sub_id]->get_image_gradients();
            mve::FloatImage::ConstPtr sub_hessian =
                this->sub_views[sub_id]->get_image_hessian();

            Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                pixels[i][0] + 0.5, pixels[i][1] + 0.5, depths[i],
                depth_derivatives[i][0], depth_derivatives[i][1]);

            C.fill(proj.begin());
            C.fill_jacobian(jac.begin());
            proj[0] -= 0.5;
            proj[1] -= 0.5;

            grad_sub[0] = sub_gradients->linear_at(proj[0], proj[1], 0);
            grad_sub[1] = sub_gradients->linear_at(proj[0], proj[1], 1);

            hess_sub[0] = sub_hessian->linear_at(proj[0], proj[1], 0);
            hess_sub[1] = sub_hessian->linear_at(proj[0], proj[1], 1);
            hess_sub[2] = hess_sub[1];
            hess_sub[3] = sub_hessian->linear_at(proj[0], proj[1], 2);

            j_grad_subs[j] = jac * grad_sub;

            C.fill_derivative(dn00, c_dn);
            C.fill_jacobian_derivative_grad(grad_sub.begin(), dn00, jac_dn);

            jac_hess = (jac * hess_sub);
            for (int col = 0; col < 16; ++col)
                jac_entries[j * 16 + col] = jac_dn[col] + jac_hess * c_dn[col];
        }

        if (this->opts.regularization > 0.0)
        {
            basic_regularizer_weight = this->opts.regularization
                // * std::max(0.03, this->grad_main.abs_sum());
                * 0.005 / std::max(0.03, this->grad_main.abs_sum());
            double x = pixels[i][0] + 0.5 -
                static_cast<double>(this->main_view->get_width()) / 2.0;
            double y = pixels[i][1] + 0.5 -
                static_cast<double>(this->main_view->get_height()) / 2.0;

            surfderiv::normal_divergence(x, y,
                this->main_view->get_flen(), depths[i],
                depth_derivatives[i][0], depth_derivatives[i][1],
                depth_2nd_derivatives[i][0], depth_2nd_derivatives[i][1],
                depth_2nd_derivatives[i][2],
                this->full_surface_div);

            surfderiv::normal_divergence_deriv(dn00, x, y,
                this->main_view->get_flen(), depths[i],
                depth_derivatives[i][0], depth_derivatives[i][1],
                depth_2nd_derivatives[i][0], depth_2nd_derivatives[i][1],
                depth_2nd_derivatives[i][2],
                this->full_surface_div_deriv);

            surfderiv::normal_derivative(dn00, x, y,
                this->main_view->get_flen(), depths[i],
                depth_derivatives[i][0], depth_derivatives[i][1],
                // depth_2nd_derivatives[i][0], depth_2nd_derivatives[i][1],
                // depth_2nd_derivatives[i][2],
                this->normal_deriv);
        }
        this->fill_gradient_and_hessian_entries(i, patch_id, gradient,
            hessian_entries);
    }
}

void
DepthOptimizer::fill_gradient_and_hessian_entries(std::size_t i,
    std::size_t patch_id, double * gradient, double * hessian_entries)
{
    double weight[2];
    std::size_t const num_diffs = (this->subsurfaces[patch_id].size() 
        * this->subsurfaces[patch_id].size() + 1) / 2;
    
#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
    __m128d reg_grad_main = _mm_set_pd(grad_main[1], grad_main[0]);
    static __m128d const sign_mask = _mm_set1_pd(-0.);
    __m128d reg_grad[16];
    __m128d reg_hessian[256];
    for (int col = 0; col < 16; ++col)
        reg_grad[col] = _mm_setzero_pd();
    for (int i = 0; i < 256; ++i)
        reg_hessian[i] = _mm_setzero_pd();
    
    for (std::size_t j = 0; j < this->subsurfaces[patch_id].size(); ++j)
    {
        __m128d reg_jgrad_sub = _mm_load_pd(*j_grad_subs[j]);
        __m128d reg_diff = _mm_sub_pd(reg_jgrad_sub, reg_grad_main);
        __m128d reg_weight = _mm_add_pd(_mm_andnot_pd(sign_mask, reg_diff),
            _mm_set1_pd(R_FACTOR));
        
        for (std::size_t j2 = j + 1; j2 < this->subsurfaces[patch_id].size();
             ++j2)
        {
            __m128d reg_jgrad_sub2 = _mm_load_pd(*j_grad_subs[j2]);
            __m128d reg_subdiff = _mm_sub_pd(reg_jgrad_sub, reg_jgrad_sub2);
            __m128d reg_subweight = _mm_add_pd(
                _mm_andnot_pd(sign_mask, reg_subdiff), _mm_set1_pd(R_FACTOR));
            for (int col = 0; col < 16; ++col)
            {
                __m128d reg_jcol = _mm_load_pd(*jac_entries[j * 16 + col]);
                __m128d reg_j2col = _mm_load_pd(*jac_entries[j2 * 16 + col]);
                __m128d reg_jace = _mm_div_pd(_mm_sub_pd(reg_jcol, reg_j2col),
                    reg_subweight);
                
                if (j2 == j + 1)
                    reg_grad[col] = _mm_add_pd(reg_grad[col],
                        _mm_div_pd(_mm_mul_pd(reg_diff, reg_jcol), reg_weight));

                reg_grad[col] = _mm_add_pd(reg_grad[col],
                    _mm_mul_pd(reg_jace, reg_subdiff));
                for (int col2 = col; col2 < 16; ++col2)
                {
                    __m128d reg_jcol2 = _mm_load_pd(
                        *jac_entries[j * 16 + col2]);
                    __m128d reg_j2col2 = _mm_load_pd(
                        *jac_entries[j2 * 16 + col2]);
                    reg_hessian[col * 16 + col2] =
                        _mm_add_pd(reg_hessian[col * 16 + col2],
                            _mm_mul_pd(reg_jace,
                                _mm_sub_pd(reg_jcol2, reg_j2col2)));
                    
                    if (j2 == j + 1)
                        reg_hessian[col * 16 + col2] =
                            _mm_add_pd(reg_hessian[col * 16 + col2],
                                _mm_mul_pd(reg_jcol,
                                    _mm_div_pd(reg_jcol2, reg_weight)));
                }
            }
        }
    }
    
    for (int col = 0; col < 16; ++col)
    {
        double const* tmp = reinterpret_cast<double const*>(&reg_grad[col]);
        gradient[col] += tmp[0] + tmp[1];
    }
    for (int col = 0; col < 16; ++col)
       for (int col2 = col; col2 < 16; ++col2)
        {
            double const* tmp = reinterpret_cast<double const*>(
                &reg_hessian[col * 16 + col2]);
            hessian_entries[col * 16 + col2] += tmp[0] + tmp[1];
        }
    
#else /* No SSE4 Support */
    double subweight[2];
    double diff[2];
    double subdiff[2];
    double jace[2];
    double jace2[2];

    for (std::size_t j = 0; j < this->subsurfaces[patch_id].size(); ++j)
    {
        diff[0] = j_grad_subs[j][0] - grad_main[0];
        diff[1] = j_grad_subs[j][1] - grad_main[1];
        weight[0] = 1.0 / (R_FACTOR + std::abs(diff[0]));
        weight[1] = 1.0 / (R_FACTOR + std::abs(diff[1]));

        for (std::size_t j2 = j + 1; j2 < this->subsurfaces[patch_id].size();
             ++j2)
        {
            subdiff[0] = j_grad_subs[j][0] - j_grad_subs[j2][0];
            subdiff[1] = j_grad_subs[j][1] - j_grad_subs[j2][1];
            subweight[0] = 1.0 / (R_FACTOR + std::abs(subdiff[0]));
            subweight[1] = 1.0 / (R_FACTOR + std::abs(subdiff[1]));

            for (int col = 0; col < 16; ++col)
            {
                jace[0] = (jac_entries[j * 16 + col][0]
                    - jac_entries[j2 * 16 + col][0]) * subweight[0];
                jace[1] = (jac_entries[j * 16 + col][1]
                    - jac_entries[j2 * 16 + col][1]) * subweight[1];

                if (j2 == j + 1)
                {
                    gradient[col] +=
                        (diff[0] * weight[0] * jac_entries[j * 16 + col][0]
                        + diff[1] * weight[1] * jac_entries[j * 16 + col][1]);
                }
                gradient[col] += (jace[0] * subdiff[0] + jace[1] * subdiff[1]);

                for (int col2 = col; col2 < 16; ++col2)
                {
                    hessian_entries[col * 16 + col2] +=
                        (jace[0] * (jac_entries[j * 16 + col2][0] -
                                    jac_entries[j2 * 16 + col2][0])
                         + jace[1] * (jac_entries[j * 16 + col2][1] -
                                      jac_entries[j2 * 16 + col2][1]));

                    if (j2 == j + 1)
                    {
                        hessian_entries[col * 16 + col2] +=
                            (jac_entries[j * 16 + col][0] * weight[0]
                             * jac_entries[j * 16 + col2][0]
                             + jac_entries[j * 16 + col][1] * weight[1]
                             * jac_entries[j * 16 + col2][1]);
                    }
                }
            }
        }
    }
#endif /* SSE3 Support */
    
    if (this->opts.regularization <= 0.0)
        return;

    /* basic regularization */
    basic_regularizer_weight *= num_diffs;
    if (this->lighting == nullptr
        || this->opts.light_surf_regularization > 0.0)
    {
        double geom_weight = 1.0;
        if (this->lighting!= nullptr)
            geom_weight *= this->opts.light_surf_regularization / 100;

        for (int v = 0; v < 6; ++v)
        {
            double const weight = geom_weight /
                (R_FACTOR + std::abs(full_surface_div[v]));
            for (int col = 0; col < 16; ++col)
            {
                gradient[col] += full_surface_div_deriv[16 * v + col]
                    * full_surface_div[v]
                    * basic_regularizer_weight * weight;

                for (int col2 = col; col2 < 16; ++col2)
                    hessian_entries[col * 16 + col2] +=
                        full_surface_div_deriv[v * 16 + col] *
                        full_surface_div_deriv[v * 16 + col2] *
                        basic_regularizer_weight * weight;
            }
        }
        if (this->lighting == nullptr)
            return;
    }

    /* shading based energy term */
    math::Vec3d normal;
    double x = pixels[i][0] + 0.5 -
        static_cast<double>(this->main_view->get_width()) / 2.0;
    double y = pixels[i][1] + 0.5 -
        static_cast<double>(this->main_view->get_height()) / 2.0;
    surfderiv::fill_normal(x, y, this->main_view->get_inverse_flen(),
        depths[i], depth_derivatives[i][0], depth_derivatives[i][1], *normal);

    GlobalLighting::Params lightparams = this->lighting->get_parameters();
    double sh_deriv[16 * 3];
    sh::derivative_4_band(*normal, sh_deriv);

    double shading = this->lighting->value_for_normal(normal);
    math::Vec2d linear_image_grad;
    linear_image_grad[0] = this->main_view->get_linear_gradients()->at(
        pixels[i][0], pixels[i][1], 0);
    linear_image_grad[1] = this->main_view->get_linear_gradients()->at(
        pixels[i][0], pixels[i][1], 1);
    double linear_image_value = this->main_view->get_linear_image()->at(
        pixels[i][0], pixels[i][1], 0);
    
    double shading_weight = 0.001 * num_diffs /
        (R_FACTOR + linear_image_grad.abs_sum());

    if(linear_image_grad.norm() < 1e-10)
        return;
    if(MATH_POW2(shading) < 1e-10 || MATH_POW2(linear_image_value) < 1e-10)
        return;

    math::Vec2d shading_grad(0.0, 0.0);
    for (int l = 1; l < 16; ++l) // sh0 is constant
    {
        shading_grad[0] += lightparams[l] * (
            sh_deriv[l * 3 + 0] * full_surface_div[0] +
            sh_deriv[l * 3 + 1] * full_surface_div[1] +
            sh_deriv[l * 3 + 2] * full_surface_div[2]);
        shading_grad[1] += lightparams[l] * (
            sh_deriv[l * 3 + 0] * full_surface_div[3] +
            sh_deriv[l * 3 + 1] * full_surface_div[4] +
            sh_deriv[l * 3 + 2] * full_surface_div[5]);
    }

    math::Vec2d render_grad = shading_grad / shading;

    linear_image_grad *= 1.0 / linear_image_value;
    math::Vec2d shading_error = render_grad - linear_image_grad;

    double shading_deriv[16];
    for (int col = 0; col < 16; ++col)
    {
        shading_deriv[col] = 0;
        for (int l = 1; l < 16; ++l) // sh0 is constant
        {
            shading_deriv[col] += lightparams[l] * (
                sh_deriv[l * 3 + 0] * normal_deriv[0 + col] +
                sh_deriv[l * 3 + 1] * normal_deriv[16 + col] +
                sh_deriv[l * 3 + 2] * normal_deriv[32 + col]);
        }
    }
    math::Vec2d shading_grad_deriv[16];
    for (int col = 0; col < 16; ++col)
    {
        shading_grad_deriv[col].fill(0.0);
        for (int l = 1; l < 16; ++l) // sh0 is constant
        {
            shading_grad_deriv[col][0] += lightparams[l] * (
                sh_deriv[l * 3 + 0] * full_surface_div_deriv[16 * 0 + col] +
                sh_deriv[l * 3 + 1] * full_surface_div_deriv[16 * 1 + col] +
                sh_deriv[l * 3 + 2] * full_surface_div_deriv[16 * 2 + col]);
            shading_grad_deriv[col][1] += lightparams[l] * (
                sh_deriv[l * 3 + 0] * full_surface_div_deriv[16 * 3 + col] +
                sh_deriv[l * 3 + 1] * full_surface_div_deriv[16 * 4 + col] +
                sh_deriv[l * 3 + 2] * full_surface_div_deriv[16 * 5 + col]);
        }
    }
    math::Vec2d render_deriv[16];
    for (int col = 0; col < 16; ++col)
        render_deriv[col] = (shading_grad_deriv[col] * shading - shading_grad *
            shading_deriv[col]) / MATH_POW2(shading);

    weight[0] = 1.0 / (R_FACTOR + std::abs(shading_error[0]));
    weight[1] = 1.0 / (R_FACTOR + std::abs(shading_error[1]));

    weight[0] *= shading_weight;
    weight[1] *= shading_weight;

    for (int col = 0; col < 16; ++col)
    {
        gradient[col] += shading_error[0] * render_deriv[col][0] * weight[0]
            + shading_error[1] * render_deriv[col][1] * weight[1];
        for (int col2 = col; col2 < 16; ++col2)
            hessian_entries[col * 16 + col2] +=
                render_deriv[col][0] * render_deriv[col2][0] * weight[0]
                + render_deriv[col][1] * render_deriv[col2][1] * weight[1];
    }
   
    return;
}

void
DepthOptimizer::reproject_neighbor (std::size_t neighbor)
{
    StereoView::Ptr neighbor_view = this->sub_views[neighbor];
#if SMVS_DEBUG
    mve::FloatImage::Ptr debug = this->main_view->get_debug_image();
    debug->fill(0);

    for (std::size_t patch_id = 0;
         patch_id < this->surface->get_patches().size(); ++patch_id)
    {
        Surface::Patch::Ptr patch = this->surface->get_patches()[patch_id];
        if (patch == nullptr)
            continue;

        patch->fill_values_at_pixels(&pixels, &depths, &depth_derivatives,
            &depth_2nd_derivatives, &pids);

        for (std::size_t i = 0; i < pixels.size(); ++i)
        {
            for (std::size_t j = 0; j < this->subsurfaces[patch_id].size(); ++j)
            {
                std::size_t sub_id = this->subsurfaces[patch_id][j];
                mve::FloatImage::ConstPtr subimage =
                    this->sub_views[sub_id]->get_image();

                Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                    pixels[i][0] + 0.5, pixels[i][1] + 0.5, depths[i],
                    depth_derivatives[i][0], depth_derivatives[i][1]);

                C.fill(proj.begin());
                C.fill_jacobian(jac.begin());
                proj[0] -= 0.5;
                proj[1] -= 0.5;

                debug->at(pixels[i][0], pixels[i][1], 0) =
                    subimage->linear_at(proj[0], proj[1], 0);
                debug->at(pixels[i][0], pixels[i][1], 1) =
                    subimage->linear_at(proj[0], proj[1], 0);
                debug->at(pixels[i][0], pixels[i][1], 2) =
                    subimage->linear_at(proj[0], proj[1], 0);
            }
        }
    }
    this->main_view->write_debug(neighbor_view->get_view_id());
#endif
}

double
DepthOptimizer::mse_for_patch (std::size_t patch_id)
{
    Surface::Patch::Ptr patch = this->surface->get_patches()[patch_id];

    patch->fill_values_at_pixels(&pixels, &depths, &depth_derivatives,
        &depth_2nd_derivatives, &pids);

    double error = 0.0;
    double counter = 0.0;
    for (std::size_t i = 0; i < pixels.size(); ++i)
    {
        this->grad_main[0] = this->main_gradients->at(
            pixels[i][0], pixels[i][1], 0);
        this->grad_main[1] = this->main_gradients->at(
            pixels[i][0], pixels[i][1], 1);

        for (std::size_t j = 0; j < this->subsurfaces[patch_id].size(); ++j)
        {
            std::size_t sub_id = this->subsurfaces[patch_id][j];
            mve::FloatImage::ConstPtr sub_gradients =
                this->sub_views[sub_id]->get_image_gradients();

            Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                pixels[i][0] + 0.5, pixels[i][1] + 0.5, depths[i],
                depth_derivatives[i][0], depth_derivatives[i][1]);

            C.fill(proj.begin());
            C.fill_jacobian(jac.begin());
            proj[0] -= 0.5;
            proj[1] -= 0.5;

            grad_sub[0] = sub_gradients->linear_at(proj[0], proj[1], 0);
            grad_sub[1] = sub_gradients->linear_at(proj[0], proj[1], 1);

            math::Vec2d diff = this->grad_main - (jac * grad_sub);
            error += diff.norm();
            if (std::isnan(error)) {
                std::cout << grad_sub[0] << " " << grad_sub[1] << std::endl;
            }
            counter += 1.0;
        }
    }
    if (counter == 0.0)
        return 1.0;
    return error / counter;
}

double
DepthOptimizer::ncc_for_patch (std::size_t patch_id, std::size_t sub_id)
{
    mve::FloatImage::ConstPtr main_image =
        this->main_view->get_image();

    std::vector<double> errors;

    Surface::Patch::Ptr patch = this->surface->get_patches()[patch_id];
    std::vector<math::Vec2d> pixels;
    std::vector<double> depths;
    patch->fill_values_at_pixels(&pixels, &depths);

    mve::FloatImage::ConstPtr sub_image =
        this->sub_views[sub_id]->get_image();

    DenseVector values0;
    DenseVector values1;
    values0.resize(pixels.size() * 3);
    values1.resize(pixels.size() * 3);

    math::Vec3d means0(0,0,0);
    math::Vec3d means1(0,0,0);
    math::Vec3d counter(0,0,0);
    math::Vec2d proj;
    math::Vec3d color_main;
    math::Vec3d color_sub;
    
    for (std::size_t i = 0; i < pixels.size(); ++i)
    {
        Correspondence C(this->Mi[sub_id], this->ti[sub_id],
            pixels[i][0] + 0.5, pixels[i][1] + 0.5, depths[i]);

        C.fill(*proj);
        proj[0] -= 0.5;
        proj[1] -= 0.5;

        if (proj[0] < 1 || proj[0] > sub_image->width() - 2
            || proj[1] < 1 || proj[1] > sub_image->height() - 2)
            return -1;

        color_main[0] = main_image->at(pixels[i][0], pixels[i][1], 0);
        color_main[1] = main_image->at(pixels[i][0], pixels[i][1], 1);
        color_main[2] = main_image->at(pixels[i][0], pixels[i][1], 2);
        color_sub[0] = sub_image->linear_at(proj[0], proj[1], 0);
        color_sub[1] = sub_image->linear_at(proj[0], proj[1], 1);
        color_sub[2] = sub_image->linear_at(proj[0], proj[1], 2);

        for (int c = 0; c < 3; ++c)
        {
            counter[c] += 1.0;
            means0[c] += (color_main[c] - means0[c]) / counter[c];
            means1[c] += (color_sub[c] - means1[c]) / counter[c];

            values0[i * 3 + c] = color_main[c];
            values1[i * 3 + c] = color_sub[c];
        }
    }
    for (std::size_t i = 0; i < values0.size() / 3; ++i)
        for (int c = 0; c < 3; ++c)
        {
            values0[i * 3 + c] -= means0[c];
            values1[i * 3 + c] -= means1[c];
        }
    double norm0 = values0.square_norm();
    double norm1 = values1.square_norm();
    if (norm0 + norm1 < 0.001)
        return 1;

    return values0.dot(values1) / (norm0 * norm1);
}

double
DepthOptimizer::tex_score_for_patch (std::size_t patch_id)
{
    mve::FloatImage::ConstPtr main_image =
        this->main_view->get_image();

    std::vector<double> errors;

    Surface::Patch::Ptr patch = this->surface->get_patches()[patch_id];
    std::vector<math::Vec2d> pixels;
    std::vector<double> depths;
    patch->fill_values_at_pixels(&pixels, &depths);

    DenseVector values;
    values.resize(pixels.size() * 3);

    math::Vec3d means(0, 0, 0);
    double counter = 0;
    for (std::size_t i = 0; i < pixels.size(); ++i)
    {
        math::Vec3d color_main;
        color_main[0] = main_image->at(pixels[i][0], pixels[i][1], 0);
        color_main[1] = main_image->at(pixels[i][0], pixels[i][1], 1);
        color_main[2] = main_image->at(pixels[i][0], pixels[i][1], 2);

        counter += 1.0;
        for (int c = 0; c < 3; ++c)
        {
            means[c] += (color_main[c] - means[c]) / counter;
            values[i * 3 + c] = color_main[c];
        }
    }
    if (means.abs_sum() < 0.05)
        return 0;

    double score = 0.0;
    for (std::size_t i = 0; i < values.size() / 3; ++i)
        for (int c = 0; c < 3; ++c)
            score += std::abs(values[i * 3 + c] - means[c]);

    return score / pixels.size();
}

SMVS_NAMESPACE_END
