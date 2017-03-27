/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>

#include "util/timer.h"
#include "mve/image_io.h"
#include "mve/image_tools.h"
#include "math/matrix_svd.h"

#include "depth_optimizer.h"
#include "correspondence.h"
#include "light_optimizer.h"
#include "conjugate_gradient.h"
#include "sgm_stereo.h"
#include "gauss_newton_step.h"

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
    int init_scale = std::max(std::ceil(std::log2(this->main_view->get_width() *
        this->main_view->get_height() / 1.7e6) / 2) + 4, 4.0);
    if (this->opts.use_sgm)
    {
        mve::FloatImage::Ptr init = this->main_view->get_sgm_depth();
        init = depthmap_bilateral_filter(init, main_view->get_image());
        if(this->opts.debug_lvl > 1)
            this->main_view->write_depth_to_view(init, "smvs-sgm-filtered");
        this->surface = Surface::create(bundle, main_view, init_scale, init);
        this->sgm_depth = init;
    }
    else
        this->surface = Surface::create(bundle, main_view, init_scale + 1);
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

        /* use shading */
        if (this->opts.use_shading && this->surface->get_scale() < 4)
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
            this->main_view->write_image_to_view(shaded, "smvs-shaded");
            mve::FloatImage::Ptr sphere =
                this->lighting->get_rendered_sphere(555);
            this->main_view->write_image_to_view(sphere, "smvs-shaded-sphere");
        }

        /* run optimization */
        this->run_newton_iterations(this->opts.num_iterations);

        if (this->opts.debug_lvl > 0)
            std::cout << "Scale " << this->surface->get_scale() << " took "
                << scale_timer.get_elapsed_sec() << "s" << std::endl;
        this->write_debug_depth();
    }

    /* Write final output */
    if (this->opts.debug_lvl > 1 && this->lighting != nullptr)
    {
        mve::FloatImage::Ptr shaded = this->lighting->render_normal_map(
            this->get_normals());
        this->main_view->write_image_to_view(shaded, "smvs-shaded");
        mve::FloatImage::Ptr sphere = this->lighting->get_rendered_sphere(555);
        this->main_view->write_image_to_view(sphere, "smvs-shaded-sphere");
        mve::FloatImage::Ptr albedo =
            this->main_view->get_linear_image()->duplicate();
        for (int p = 0; p < albedo->get_pixel_amount(); ++p)
            if (shaded->at(p) > 0.0)
               for(int c = 0; c < albedo->channels(); ++c)
                  albedo->at(p, c) /= shaded->at(p);
            else
               for(int c = 0; c < albedo->channels(); ++c)
                   albedo->at(p, c) = 0.0;
        this->main_view->write_image_to_view(albedo, "smvs-implicit-albedo");
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
    std::vector<std::pair<std::size_t, math::Vec2d>> projections1;
    std::vector<std::pair<std::size_t, math::Vec2d>> projections2;

    this->main_gradients = this->main_view->get_image_gradients();

    bool finished = false;
    for (int iter = 0; iter < num_iters; ++iter)
    {
        std::size_t num_valid_patches = 0;
        for (auto p : this->surface->get_patches())
            if (p != nullptr)
                num_valid_patches++;
        if (this->opts.debug_lvl > 0 && iter == 0)
            std::cout << "Surface Status - "
                "Valid patches: " << num_valid_patches << std::endl;

        if (iter == 0)
        {
            this->create_subview_surfaces();
            int deleted = std::numeric_limits<int>::max();
            while (deleted > 10)
                deleted = this->cut_boundaries();
        }

        GaussNewtonStep::Options gauss_newton_opts;
        gauss_newton_opts.regularization = this->opts.regularization;
        gauss_newton_opts.light_surf_regularization =
            this->opts.light_surf_regularization;
        GaussNewtonStep gauss_newton_step(gauss_newton_opts, this->main_view,
            this->sub_views, this->Mi, this->ti);

        std::size_t num_initial_active_nodes = 0;
        std::vector<char> active_nodes(this->surface->get_nodes().size(), 0);
        for (std::size_t i = 0; i < this->surface->get_nodes().size(); ++i)
            if (this->surface->get_nodes()[i] != nullptr)
            {
                active_nodes[i] = 1;
                num_initial_active_nodes += 1;
            }
        std::size_t num_active_nodes = num_initial_active_nodes;

        unsigned int newton_step = 0;
        unsigned int linear_iterations = 0;
        float timer_build_step = 0.0;
        float timer_solve_step = 0.0;

        for (; newton_step < 200
             &&  num_active_nodes > num_initial_active_nodes / 20;)
        {
            if (this->opts.debug_lvl > 1)
                std::cout << "Num active nodes: "
                    << num_active_nodes << std::endl;

            newton_step += 1;
            prev_gradient = gradient;
            util::WallTimer newton_timer;

            /* construct newton step */
            gauss_newton_step.construct(this->surface, this->subsurfaces,
                active_nodes, this->lighting, &hessian, &gradient, &precond);
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
            this->fill_node_reprojections(active_nodes, &projections1);
            this->surface->update_nodes(delta, &depth_updates);
            this->fill_node_reprojections(active_nodes, &projections2);

            if (this->opts.full_optimization)
            {
                double sum_diff = 0;
                for (std::size_t p = 0; p < projections1.size(); ++p)
                    sum_diff += (projections1[p].second -
                        projections2[p].second).norm();

                double update = sum_diff / (double) projections1.size();
                if (this->opts.debug_lvl > 2)
                    std::cout << "Avg delta: " << update << std::endl;
                if (update < 0.01)
                    break;
                else
                    continue;
            }

            std::fill(active_nodes.begin(), active_nodes.end(), 0);
            for (std::size_t p = 0; p < projections1.size(); ++p)
            {
                double diff = (projections1[p].second -
                    projections2[p].second).norm();
                if (diff > 0.15)
                    active_nodes[projections1[p].first] = 1;
            }

            num_active_nodes = 0;
            for (auto & node : active_nodes)
                if (node == 1)
                    num_active_nodes += 1;
        }

        if (this->opts.debug_lvl > 0)
        {
            std::cout << "### Finished iteration: " << iter << std::endl;
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

        /* Stop here if number of surface patches is converged */
        if (finished)
            break;

        /*  Else expand surface patches and cut boundaries */
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
        this->surface->remove_isolated_patches();

        std::size_t num_valid_new = 0;
        for (auto p : this->surface->get_patches())
            if (p != nullptr)
                num_valid_new++;

        /* check surface changes for convergence */
        double change = 1.0 - (double)std::min(num_valid_new, num_valid_patches)
            / (double)std::max(num_valid_new, num_valid_patches);

        if (this->opts.debug_lvl > 0)
            std::cout << "Surface change: " << change << " Valid patches: "
                << num_valid_patches << " -> " << num_valid_new << std::endl;
        this->write_debug_depth();

        if (iter > 0 && (num_valid_new <= num_valid_patches ||
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

        double dd_factor = 5.0;
        auto iter_first = depth_p.begin();
        auto iter_last = std::prev(depth_p.end());

        if (iter_first->second + iter_last->second == 3)
            dd_factor *= MATH_SQRT2;
        math::Vec3f v = invproj * math::Vec3f
            ((float)pixels[0][0] + 0.5f, (float)pixels[0][1] + 0.5f, 1.0f);
        double threshold = dd_factor * iter_first->first * invproj[0]
            * this->surface->get_patchsize() / v.norm();

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
        for (std::size_t node = 0; node < 4; ++node)
        {
            Surface::NodeList node_neighbors;
            this->surface->fill_node_neighbors(node_ids[node], &node_neighbors);
            int num_invalid = 0;
            for (auto neighbor : node_neighbors)
                if (neighbor == nullptr)
                    num_invalid += 1;
            if (num_invalid > 1)
                if (error > 0.05)
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

    this->pixels.clear();
    this->depths.clear();
    mve::FloatImage::Ptr depth = this->get_depth();
    for (int x = 0; x < depth->width(); ++x)
        for (int y = 0; y < depth->height(); ++y)
        {
            if (depth->at(x, y, 0) != 0)
            {
                pixels.push_back(math::Vec2d(x, y));
                depths.push_back(depth->at(x, y, 0));
            }
            if (this->opts.use_sgm && this->sgm_depth->at(x, y, 0) != 0)
            {
                pixels.push_back(math::Vec2d(x, y));
                depths.push_back(this->sgm_depth->at(x, y, 0));
            }
        }

    /* first pass: find minimal depth */
    Correspondence C;
    for (std::size_t sub_id = 0; sub_id < this->sub_views.size(); ++sub_id)
    {
        double const sub_width = static_cast<double>(
            this->sub_views[sub_id]->get_width());
        double const sub_height = static_cast<double>(
            this->sub_views[sub_id]->get_height());

        for (std::size_t i = 0; i < pixels.size(); i++)
        {
            C.update(this->Mi[sub_id], this->ti[sub_id],
                pixels[i][0] + 0.5, pixels[i][1] + 0.5, depths[i]);
            math::Vec2d proj;
            C.fill(*proj);
            proj[0] -= 0.5;
            proj[1] -= 0.5;
            double const cutoffset = 3.0;
            if (proj[0] < cutoffset || proj[0] >= sub_width - cutoffset ||
                proj[1] < cutoffset || proj[1] >= sub_height - cutoffset)
                continue;

            int cx = static_cast<int>(proj[0]);
            int cy = static_cast<int>(proj[1]);
            for (int x = -1; x < 2; ++x)
                for (int y = -1; y < 2; ++y)
            if (C.get_depth() < depth_caches[sub_id]->at(cx+x, cy+y, 0))
                depth_caches[sub_id]->at(cx+x, cy+y, 0) = C.get_depth();
        }
    }

    /* second pass: keep ids near minimal depth */
    for (std::size_t patch_id = 0; patch_id < patches.size(); ++patch_id)
    {
        if (patches[patch_id] == nullptr)
            continue;

        patches[patch_id]->fill_values_at_pixels(&pixels, &depths);

        for (std::size_t sub_id = 0; sub_id < this->sub_views.size(); ++sub_id)
        {
            double const sub_width = static_cast<double>(
                this->sub_views[sub_id]->get_width());
            double const sub_height = static_cast<double>(
                this->sub_views[sub_id]->get_height());

            bool success = true;
            for (std::size_t i = 0; i < pixels.size() && success; i++)
            {
                Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                    pixels[i][0] + 0.5, pixels[i][1] + 0.5, depths[i]);
                math::Vec2d proj;
                C.fill(*proj);
                proj[0] -= 0.5;
                proj[1] -= 0.5;
                double const cutoffset = 0.03 * std::max(sub_width, sub_height);
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
                    if (C.get_depth() * 0.95 >
                        depth_caches[sub_id]->at(cx + x, cy + y, 0))
                        success = false;
            }
            if (!success)
                continue;

            std::vector<math::Vec2d> node_coords;
            std::vector<double> node_depths;
            std::vector<math::Vec2d> node_depths_derivs;
            patches[patch_id]->fill_values_at_nodes(&node_coords,
                &node_depths, &node_depths_derivs);

            patches[patch_id]->fill_values_at_pixels(&pixels, &depths,
                &depth_derivatives);

            double max = 0.0;
            for (std::size_t i = 0; i < pixels.size(); ++i)
            {
                Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                     pixels[i][0] + 0.5, pixels[i][1] + 0.5, depths[i],
                     depth_derivatives[i][0], depth_derivatives[i][1]);
                math::Matrix2d jac;
                C.fill_jacobian(*jac);
                /* Find singular values of jacobian */
                double S[2];
                S[0] = (std::sqrt(MATH_POW2(jac[0] - jac[3])
                    + MATH_POW2(jac[1] + jac[2]))
                    + std::sqrt(MATH_POW2(jac[0] + jac[3])
                    + MATH_POW2(jac[1] - jac[2]))) / 2.0;
                S[1] = std::fabs(S[0] - std::sqrt(MATH_POW2(jac[0] - jac[3])
                    + MATH_POW2(jac[1] + jac[2])));
                double sigma0 = MATH_POW2(std::max(S[0], S[1]));
                double sigma1 = MATH_POW2(std::min(S[0], S[1]));
                max = std::max(max, sigma0 / sigma1);
            }
            if (max > 8.0)
                continue;

            /* filter possible occlusions from unreconstructed geometry */
            if (!this->opts.use_sgm &&
                this->ncc_for_patch(patch_id, sub_id) < 0)
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
DepthOptimizer::fill_node_reprojections(std::vector<char> const& active_nodes,
    std::vector<std::pair<std::size_t, math::Vec2d>> * proj)
{
    proj->clear();
    Surface::PatchList const& patches = this->surface->get_patches();
    for (int patch_id = 0; patch_id < (int)patches.size(); ++patch_id)
    {
        SurfacePatch::Ptr patch = patches[patch_id];
        if (patch == nullptr)
            continue;
        std::size_t node_ids[4];
        this->surface->fill_node_ids_for_patch(patch_id, node_ids);
        if ((active_nodes[node_ids[0]] + active_nodes[node_ids[1]]
            + active_nodes[node_ids[2]] + active_nodes[node_ids[3]]) == 0)
            continue;

        patch->fill_values_at_pixels(&pixels, &depths);
        for (std::size_t j = 0; j < this->subsurfaces[patch_id].size(); ++j)
            for (std::size_t i = 0; i < pixels.size(); ++i)
            {
                std::size_t sub_id = this->subsurfaces[patch_id][j];
                Correspondence C(this->Mi[sub_id], this->ti[sub_id],
                    pixels[i][0], pixels[i][1], depths[i]);
                math::Vec2d projection;
                C.fill(*projection);
                for (std::size_t n = 0; n < 4; ++n)
                    proj->emplace_back(node_ids[n], projection);
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
DepthOptimizer::reproject_neighbor (std::size_t neighbor)
{
    StereoView::Ptr neighbor_view = this->sub_views[neighbor];
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
    Surface::Patch::Ptr patch = this->surface->get_patches()[patch_id];

    mve::FloatImage::ConstPtr main_image =
        this->main_view->get_image();
    mve::FloatImage::ConstPtr sub_image =
        this->sub_views[sub_id]->get_image();

    /* Get corners and pixels */
    std::vector<math::Vec2d> corners;
    std::vector<double> corner_depths;
    patch->fill_values_at_nodes(&corners, &corner_depths);
    math::Vec2d min = corners[0];
    math::Vec2d max = corners[3];
    patch->fill_values_at_pixels(&pixels, &depths);

    /* Add boundary */
    if (min[0] > 1 && max[0] < main_image->width() - 2
        && min[1] > 1 && max[1] < main_image->height() - 2)
    {
        pixels.emplace_back(corners[0][0] - 1, corners[0][1] - 1);
        depths.emplace_back(corner_depths[0]);
        pixels.emplace_back(corners[1][0] + 1, corners[1][1] - 1);
        depths.emplace_back(corner_depths[1]);
        pixels.emplace_back(corners[2][0] - 1, corners[2][1] + 1);
        depths.emplace_back(corner_depths[2]);
        pixels.emplace_back(corners[3][0] + 1, corners[0][1] + 1);
        depths.emplace_back(corner_depths[3]);
    }
    for (std::size_t i = 0; i < pixels.size(); ++i)
    {
        /* top */
        if (min[1] > 2 && pixels[i][1] == min[1])
        {
            pixels.emplace_back(pixels[i][0], pixels[i][1] - 2);
            pixels.emplace_back(pixels[i][0], pixels[i][1] - 1);
            depths.emplace_back(depths[i]);
        }
        /* bottom */
        if (max[1] < main_image->height() - 3 && pixels[i][1] == max[1])
        {
            pixels.emplace_back(pixels[i][0], pixels[i][1] + 2);
            pixels.emplace_back(pixels[i][0], pixels[i][1] + 1);
            depths.emplace_back(depths[i]);
        }
        /* left */
        if (min[0] > 2 && pixels[i][0] == min[0])
        {
            pixels.emplace_back(pixels[i][0] - 2, pixels[i][1]);
            pixels.emplace_back(pixels[i][0] - 1, pixels[i][1]);
            depths.emplace_back(depths[i]);
        }
        /* right */
        if (max[0] < main_image->width() - 3 && pixels[i][0] == max[0])
        {
            pixels.emplace_back(pixels[i][0] + 2, pixels[i][1]);
            pixels.emplace_back(pixels[i][0] + 1, pixels[i][1]);
            depths.emplace_back(depths[i]);
        }
    }

    DenseVector values0;
    DenseVector values1;
    values0.resize(pixels.size() * 3);
    values1.resize(pixels.size() * 3);

    math::Vec3d means0(0,0,0);
    math::Vec3d means1(0,0,0);
    math::Vec3d counter(0,0,0);
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
    double norm0 = values0.norm();
    double norm1 = values1.norm();
    if (norm0 + norm1 < 0.001 * pixels.size())
        return 1;
    double ncc = values0.dot(values1) / (norm0 * norm1);
    return ncc;
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

mve::FloatImage::Ptr
DepthOptimizer::depthmap_bilateral_filter(mve::FloatImage::ConstPtr dm,
    mve::FloatImage::ConstPtr ci, float sigma, int kernel_size)
{
    int const dm_w(dm->width());
    int const dm_h(dm->height());

    int const w = ci->width();
    int const h = ci->height();
    mve::FloatImage::Ptr out = mve::FloatImage::create(w, h, 1);
    out->fill(0);

    float const scale_x = static_cast<float>(dm_w) / static_cast<float>(w);
    float const scale_y = static_cast<float>(dm_h) / static_cast<float>(h);

    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
        {
            math::Accum<float> accum(0.0f);
            for (int ky = -kernel_size; ky <= kernel_size; ++ky)
                for (int kx = -kernel_size; kx <= kernel_size; ++kx)
                {
                    int const ci_x = math::clamp(x + kx, 0, w - 1);
                    int const ci_y = math::clamp(y + ky, 0, h - 1);
                    int const dm_x = math::clamp(scale_x * ci_x, 0.f,
                        static_cast<float>(dm_w) - 1.f);
                    int const dm_y = math::clamp(scale_y * ci_y, 0.f,
                        static_cast<float>(dm_h) - 1.f);

                    if(dm->at(dm_x, dm_y, 0) == 0.0f)
                        continue;

                    float weight(1.0f);
                    /* spatial weight */
                    weight *= math::gaussian_2d((float)kx, (float)ky,
                        sigma, sigma);
                    /* depth value difference weight */
                    for (int c = 0; c < ci->channels(); ++c)
                        weight *= math::gaussian(ci->at(ci_x, ci_y, c)
                            - ci->at(x, y, c), 0.1f);

                    accum.add(dm->at(dm_x, dm_y, 0), weight);
                }
            if (accum.w > 0)
                out->at(x, y, 0) = accum.normalized();
        }
    return out;
}

SMVS_NAMESPACE_END
