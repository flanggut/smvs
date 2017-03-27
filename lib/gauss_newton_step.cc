/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <smmintrin.h> // SSE4_1

#include "gauss_newton_step.h"
#include "surface_derivative.h"
#include "conjugate_gradient.h"

// minimal value for reweighting linear system to optimize L1 norm
#define R_FACTOR 1e-4

SMVS_NAMESPACE_BEGIN

GaussNewtonStep::GaussNewtonStep (Options const& opts,
        StereoView::ConstPtr main_view,
        std::vector<StereoView::Ptr> const& sub_views,
        std::vector<math::Matrix3d> const& Mi,
        std::vector<math::Vec3d> const& ti)
    : opts(opts), main_view(main_view), sub_views(sub_views)
    , Mi(Mi), ti(ti)
{
    this->main_gradients = this->main_view->get_image_gradients();
    this->main_gradients_linear = this->main_view->get_shading_gradients();
}

void
GaussNewtonStep::construct (Surface::Ptr surface,
        std::vector<std::vector<std::size_t>> const& subsurfaces,
        std::vector<char> const& active_nodes,
        GlobalLighting::Ptr lighting,
        SparseMatrix * hessian, DenseVector * gradient, SparseMatrix * precond)
{
    this->lighting = lighting;
    Surface::PatchList const & patches = surface->get_patches();
    std::size_t const num_params = surface->get_num_nodes() * 4;
    int const pixels_per_patch = MATH_POW2(surface->get_patchsize());
    std::vector<double> node_derivatives;
    node_derivatives.resize(pixels_per_patch * 96);
    for (int i = 0; i < pixels_per_patch; ++i)
        surface->fill_node_derivatives_for_pixel(i,
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

    for (int patch_id = 0; patch_id < (int)surface->get_patches().size();
         ++patch_id)
    {
        SurfacePatch::Ptr patch = patches[patch_id];
        if (patch == nullptr)
            continue;
        std::size_t node_ids[4];
        surface->fill_node_ids_for_patch(patch_id, node_ids);

        if (active_nodes[node_ids[0]] == 0
            && active_nodes[node_ids[1]] == 0
            && active_nodes[node_ids[2]] == 0
            && active_nodes[node_ids[3]] == 0)
        {
            continue;
        }

        std::fill(sub_gradient, sub_gradient + 16, 0.0);
        std::fill(sub_hessian, sub_hessian + 256, 0.0);

        this->jacobian_entries_for_patch(surface->get_scale(),
            patch, subsurfaces[patch_id],
            node_derivatives, sub_gradient, sub_hessian);

        /* compute gradient block */
        for (int node = 0; node < 4; ++node)
        {
            if (active_nodes[node_ids[node]] == 0)
                continue;
            for (int value = 0; value < 4; ++value)
                gradient->at(node_ids[node] * 4 + value) +=
                    sub_gradient[node * 4 + value];
        }

        /* compute hessian block */
        for (std::size_t node1 = 0; node1 < 16; ++node1)
        {
            if (active_nodes[node_ids[node1 / 4]] == 0)
                continue;
            for (std::size_t node2 = node1; node2 < 16; ++node2)
            {
                if (active_nodes[node_ids[node2 / 4]] == 0)
                    continue;
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
    }
    hessian->allocate(num_params, num_params);
    precond->allocate(num_params, num_params);
    
    SparseMatrix::Blocks hessian_block_vec;
    hessian_block_vec.reserve(surface->get_num_nodes() * 9);
    SparseMatrix::Blocks precond_block_vec;
    precond_block_vec.reserve(surface->get_num_nodes());
    
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
GaussNewtonStep::jacobian_entries_for_patch(int const scale,
    Surface::Patch::Ptr patch,
    std::vector<std::size_t> const& patch_neighbors,
    std::vector<double> const& node_derivatives, double * gradient,
    double * hessian_entries)
{
    std::size_t num_sds = patch_neighbors.size() + 1;
    num_sds = num_sds * (num_sds + 1);
    p_weights.resize(num_sds);
    p_diffs.resize(num_sds);

    int sampling = 4;
    if (scale < 5)
        sampling = 2;
    if (scale < 3)
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

        for (std::size_t j = 0; j < patch_neighbors.size(); ++j)
        {
            std::size_t sub_id = patch_neighbors[j];
            mve::FloatImage::ConstPtr sub_gradients =
                this->sub_views[sub_id]->get_image_gradients();
            mve::FloatImage::ConstPtr sub_hessian =
                this->sub_views[sub_id]->get_image_hessian();

            C.update(this->Mi[sub_id], this->ti[sub_id],
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
        this->fill_gradient_and_hessian_entries(i, patch_neighbors.size(),
            gradient, hessian_entries);
    }
}

void
GaussNewtonStep::fill_gradient_and_hessian_entries(std::size_t i,
    std::size_t num_subs, double * gradient, double * hessian_entries)
{
    double weight[2];
    
#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
    __m128d reg_grad_main = _mm_set_pd(grad_main[1], grad_main[0]);
    static __m128d const sign_mask = _mm_set1_pd(-0.);
    static __m128d const reg_rfactor = _mm_set1_pd(R_FACTOR);

    reg_grad_mem.resize(16);
    reg_hessian_mem.resize(256);

    __m128d * reg_grad = reinterpret_cast<__m128d *>(*reg_grad_mem[0]);
    __m128d * reg_hessian = reinterpret_cast<__m128d *>(*reg_hessian_mem[0]);

    for (int col = 0; col < 16; ++col)
        reg_grad[col] = _mm_setzero_pd();
    for (int i = 0; i < 256; ++i)
        reg_hessian[i] = _mm_setzero_pd();

    for (std::size_t j = 0; j < num_subs; ++j)
    {
        __m128d reg_jgrad_sub = _mm_load_pd(*j_grad_subs[j]);
        __m128d reg_diff = _mm_sub_pd(reg_jgrad_sub, reg_grad_main);
        __m128d reg_weight = _mm_add_pd(_mm_andnot_pd(sign_mask, reg_diff),
            reg_rfactor);

        __m128d * reg_jcol = reinterpret_cast<__m128d *>(*jac_entries[j * 16]);
        for (int col = 0; col < 16; ++col, ++reg_jcol)
        {
            reg_grad[col] = _mm_add_pd(reg_grad[col], _mm_div_pd(
                _mm_mul_pd(reg_diff, *reg_jcol), reg_weight));
            __m128d * reg_jcol2 = reinterpret_cast<__m128d *>(
                *jac_entries[j * 16 + col]);

            for (int col2 = col; col2 < 16; ++col2, ++reg_jcol2)
                reg_hessian[col * 16 + col2] =
                    _mm_add_pd(reg_hessian[col * 16 + col2],
                        _mm_mul_pd(*reg_jcol,
                            _mm_div_pd(*reg_jcol2, reg_weight)));
        }

        for (std::size_t j2 = j + 1; j2 < num_subs; ++j2)
        {
            __m128d reg_jgrad_sub2 = _mm_load_pd(*j_grad_subs[j2]);
            __m128d reg_subdiff = _mm_sub_pd(reg_jgrad_sub, reg_jgrad_sub2);
            __m128d reg_subweight = _mm_add_pd(
                _mm_andnot_pd(sign_mask, reg_subdiff), reg_rfactor);

            __m128d * reg_jcol = reinterpret_cast<__m128d *>(
                *jac_entries[j * 16]);
            __m128d * reg_j2col = reinterpret_cast<__m128d *>(
                *jac_entries[j2 * 16]);
            for (int col = 0; col < 16; ++col, ++reg_jcol, ++reg_j2col)
            {
                __m128d reg_jace = _mm_div_pd(_mm_sub_pd(*reg_jcol, *reg_j2col),
                     reg_subweight);

                reg_grad[col] = _mm_add_pd(reg_grad[col],
                    _mm_mul_pd(reg_jace, reg_subdiff));

                __m128d * reg_jcol2 = reinterpret_cast<__m128d *>(
                    *jac_entries[j * 16 + col]);
                __m128d * reg_j2col2 = reinterpret_cast<__m128d *>(
                    *jac_entries[j2 * 16 + col]);
                for (int col2 = col; col2 < 16; ++col2,
                     ++reg_jcol2, ++reg_j2col2)
                    reg_hessian[col * 16 + col2] =
                        _mm_add_pd(reg_hessian[col * 16 + col2],
                            _mm_mul_pd(reg_jace,
                                _mm_sub_pd(*reg_jcol2, *reg_j2col2)));
            }
        }
    }
    
    for (int col = 0; col < 16; ++col)
        gradient[col] += reg_grad_mem[col][0] + reg_grad_mem[col][1];

    for (int col = 0; col < 16; ++col)
       for (int col2 = col; col2 < 16; ++col2)
        {
            hessian_entries[col * 16 + col2] +=
                reg_hessian_mem[col * 16 + col2][0] +
                reg_hessian_mem[col * 16 + col2][1];
        }
    
#else /* No SSE4 Support */
    double subweight[2];
    double diff[2];
    double subdiff[2];
    double jace[2];

    for (std::size_t j = 0; j < num_subs; ++j)
    {
        diff[0] = j_grad_subs[j][0] - grad_main[0];
        diff[1] = j_grad_subs[j][1] - grad_main[1];
        weight[0] = 1.0 / (R_FACTOR + std::abs(diff[0]));
        weight[1] = 1.0 / (R_FACTOR + std::abs(diff[1]));

        for (int col = 0; col < 16; ++col)
        {
            gradient[col] += (diff[0] * weight[0] * jac_entries[j * 16 + col][0]
                 + diff[1] * weight[1] * jac_entries[j * 16 + col][1]);
            for (int col2 = col; col2 < 16; ++col2)
                hessian_entries[col * 16 + col2] +=
                    (jac_entries[j * 16 + col][0] * weight[0]
                     * jac_entries[j * 16 + col2][0]
                     + jac_entries[j * 16 + col][1] * weight[1]
                     * jac_entries[j * 16 + col2][1]);
        }

        for (std::size_t j2 = j + 1; j2 < num_subs; ++j2)
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
                gradient[col] += (jace[0] * subdiff[0] + jace[1] * subdiff[1]);

                for (int col2 = col; col2 < 16; ++col2)
                    hessian_entries[col * 16 + col2] +=
                        (jace[0] * (jac_entries[j * 16 + col2][0] -
                                    jac_entries[j2 * 16 + col2][0])
                         + jace[1] * (jac_entries[j * 16 + col2][1] -
                                      jac_entries[j2 * 16 + col2][1]));
            }
        }
    }
#endif /* SSE3 Support */
    
    if (this->opts.regularization <= 0.0)
        return;

    std::size_t const num_diffs = (num_subs * (num_subs + 1)) / 2;

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
    linear_image_grad[0] = this->main_gradients_linear->at(
        pixels[i][0], pixels[i][1], 0);
    linear_image_grad[1] = this->main_gradients_linear->at(
        pixels[i][0], pixels[i][1], 1);
    double linear_image_value = this->main_view->get_shading_image()->at(
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

SMVS_NAMESPACE_END
