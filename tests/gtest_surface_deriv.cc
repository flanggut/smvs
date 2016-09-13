/*
 * Copyright (C) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <gtest/gtest.h>

#include "math/matrix.h"
#include "math/vector.h"

#include "bicubic_patch.h"
#include "surface_derivative.h"

using namespace smvs;

void
compare_curvature_derivative(double deriv, double base_value,
                             double delta, BicubicPatch::ConstPtr patch_2)
{
    double const patch_to_pixel = 0.2;
    double dx_2 = patch_2->evaluate_dx(0.7, 0.1) * patch_to_pixel;
    double dy_2 = patch_2->evaluate_dy(0.7, 0.1) * patch_to_pixel;
    double dxy_2 = patch_2->evaluate_dxy(0.7, 0.1) * patch_to_pixel;
    double dxx_2 = patch_2->evaluate_dxx(0.7, 0.1) * patch_to_pixel;
    double dyy_2 = patch_2->evaluate_dyy(0.7, 0.1) * patch_to_pixel;

    double new_value = surfderiv::mean_curvature(dx_2, dy_2, dxy_2, dxx_2, dyy_2);

    EXPECT_NEAR(deriv, (new_value - base_value) / delta, 1e-4);
}

TEST(MeanCurvatureTest, Derivatives)
{
    double const patch_to_pixel = 0.2;

    double d_00[96];
    BicubicPatch::node_derivatives(0.7, 0.1, d_00, d_00 + 24,
                                   d_00 + 48, d_00 + 72);

    for (int n = 0; n < 4; ++n)
        for (int i = 4; i < 24; ++i)
            d_00[24 * n + i] *= patch_to_pixel;

    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 10.0; n00->dx = 4.0; n00->dy = 4.0; n00->dxy = -8.0;
    n10->f = 10.0; n10->dx = -4.0; n10->dy = 4.0; n10->dxy = -8.0;
    n01->f = 10.0; n01->dx = 4.0; n01->dy = -4.0; n01->dxy = -8.0;
    n11->f = 10.0; n11->dx = -4.0; n11->dy = -4.0; n11->dxy = -8.0;

    BicubicPatch::Ptr patch_1 = BicubicPatch::create(n00, n10, n01, n11);
    BicubicPatch::Ptr patch_2;

    double dx_1 = patch_1->evaluate_dx(0.7, 0.1) * patch_to_pixel;
    double dy_1 = patch_1->evaluate_dy(0.7, 0.1) * patch_to_pixel;
    double dxy_1 = patch_1->evaluate_dxy(0.7, 0.1) * patch_to_pixel;
    double dxx_1 = patch_1->evaluate_dxx(0.7, 0.1) * patch_to_pixel;
    double dyy_1 = patch_1->evaluate_dyy(0.7, 0.1) * patch_to_pixel;

    double base_value = surfderiv::mean_curvature(dx_1, dy_1,
        dxy_1, dxx_1, dyy_1);

    double const delta = 1e-5;
    double deriv[16];
    surfderiv::mean_curvature_derivative(d_00, dx_1, dy_1, dxy_1, dxx_1, dyy_1, deriv);

    double backup;

    /* Node 00 */
    backup = n00->f;
    n00->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[0], base_value, delta, patch_2);
    n00->f = backup;

    backup = n00->dx;
    n00->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[1], base_value, delta, patch_2);
    n00->dx = backup;

    backup = n00->dy;
    n00->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[2], base_value, delta, patch_2);
    n00->dy = backup;

    backup = n00->dxy;
    n00->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[3], base_value, delta, patch_2);
    n00->dxy = backup;

    /* Node 10 */
    backup = n10->f;
    n10->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[4], base_value, delta, patch_2);
    n10->f = backup;

    backup = n10->dx;
    n10->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[5], base_value, delta, patch_2);
    n10->dx = backup;

    backup = n10->dy;
    n10->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[6], base_value, delta, patch_2);
    n10->dy = backup;

    backup = n10->dxy;
    n10->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[7], base_value, delta, patch_2);
    n10->dxy = backup;

    /* Node 01 */
    backup = n01->f;
    n01->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[8], base_value, delta, patch_2);
    n01->f = backup;

    backup = n01->dx;
    n01->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[9], base_value, delta, patch_2);
    n01->dx = backup;

    backup = n01->dy;
    n01->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[10], base_value, delta, patch_2);
    n01->dy = backup;

    backup = n01->dxy;
    n01->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[11], base_value, delta, patch_2);
    n01->dxy = backup;

    /* Node 11 */
    backup = n11->f;
    n11->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[12], base_value, delta, patch_2);
    n11->f = backup;

    backup = n11->dx;
    n11->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[13], base_value, delta, patch_2);
    n11->dx = backup;

    backup = n11->dy;
    n11->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[14], base_value, delta, patch_2);
    n11->dy = backup;

    backup = n11->dxy;
    n11->dxy += delta ;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_curvature_derivative(deriv[15], base_value, delta, patch_2);
    n11->dxy = backup;
}


void
compare_normal_derivative (double delta,
    BicubicPatch::ConstPtr patch_2, math::Vec3d const& base_normal, int id,
    double * analytic)
{
    double p_x = 0.7;
    double p_y = 0.2;
    double x = 100;
    double y = 200;
    double f = 500;
    double inv_f = 1.0 / f;
    double const patch_to_pixel = 0.2;

    double w_2 = patch_2->evaluate_f(p_x, p_y);
    double dx_2 = patch_2->evaluate_dx(p_x, p_y) * patch_to_pixel;
    double dy_2 = patch_2->evaluate_dy(p_x, p_y) * patch_to_pixel;

    math::Vec3d normal_2;
    surfderiv::fill_normal(x, y, inv_f, w_2, dx_2, dy_2, *normal_2);

    math::Vec3d numeric_deriv = (normal_2 - base_normal);
    numeric_deriv /= delta;

    EXPECT_NEAR(analytic[0 + id], numeric_deriv[0], 1e-5);
    EXPECT_NEAR(analytic[16 + id], numeric_deriv[1], 1e-5);
    EXPECT_NEAR(analytic[32 + id], numeric_deriv[2], 1e-5);
}



TEST(NormalDerivativeTest, Values)
{
    double p_x = 0.7;
    double p_y = 0.2;

    double const patch_to_pixel = 0.2;

    double d_00[96];
    BicubicPatch::node_derivatives(p_x, p_y, d_00, d_00 + 24,
        d_00 + 48, d_00 + 72);

    for (int n = 0; n < 4; ++n)
    {
        for (int i = 4; i < 24; ++i)
            d_00[24 * n + i] *= patch_to_pixel;
        for (int i = 12; i < 24; ++i)
            d_00[24 * n + i] *= patch_to_pixel;
    }

    double x = 100;
    double y = 200;
    double f = 500;
    double inv_f = 1.0 / f;

    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 10.0; n00->dx = 4.0; n00->dy = 4.0; n00->dxy = -8.0;
    n10->f = 10.0; n10->dx = -4.0; n10->dy = 4.0; n10->dxy = -8.0;
    n01->f = 10.0; n01->dx = 4.0; n01->dy = -4.0; n01->dxy = -8.0;
    n11->f = 10.0; n11->dx = -4.0; n11->dy = -4.0; n11->dxy = -8.0;

    BicubicPatch::Ptr patch_1 = BicubicPatch::create(n00, n10, n01, n11);
    BicubicPatch::Ptr patch_2;

    double w_1 = patch_1->evaluate_f(p_x, p_y);
    double dx_1 = patch_1->evaluate_dx(p_x, p_y) * patch_to_pixel;
    double dy_1 = patch_1->evaluate_dy(p_x, p_y) * patch_to_pixel;

    math::Vec3d base_normal;
    surfderiv::fill_normal(x, y, inv_f, w_1, dx_1, dy_1, *base_normal);

    double analytic_full_deriv[48];
    surfderiv::normal_derivative(d_00, x, y, f, w_1, dx_1, dy_1,
        analytic_full_deriv);

    double const delta = 1e-5;
    double backup;

    /* Node 00 */
    backup = n00->f;
    n00->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 0,
        analytic_full_deriv);
    n00->f = backup;

    backup = n00->dx;
    n00->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 1,
        analytic_full_deriv);
    n00->dx = backup;

    backup = n00->dy;
    n00->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 2,
        analytic_full_deriv);
    n00->dy = backup;

    backup = n00->dxy;
    n00->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 3,
        analytic_full_deriv);
    n00->dxy = backup;

    /* Node 10 */
    backup = n10->f;
    n10->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 4,
        analytic_full_deriv);
    n10->f = backup;

    backup = n10->dx;
    n10->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 5,
        analytic_full_deriv);
    n10->dx = backup;

    backup = n10->dy;
    n10->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 6,
        analytic_full_deriv);
    n10->dy = backup;

    backup = n10->dxy;
    n10->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 7,
        analytic_full_deriv);
    n10->dxy = backup;

    /* Node 01 */
    backup = n01->f;
    n01->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 8,
        analytic_full_deriv);
    n01->f = backup;

    backup = n01->dx;
    n01->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 9,
        analytic_full_deriv);
    n01->dx = backup;

    backup = n01->dy;
    n01->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 10,
        analytic_full_deriv);
    n01->dy = backup;

    backup = n01->dxy;
    n01->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 11,
        analytic_full_deriv);
    n01->dxy = backup;

    /* Node 11 */
    backup = n11->f;
    n11->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 12,
        analytic_full_deriv);
    n11->f = backup;

    backup = n11->dx;
    n11->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 13,
        analytic_full_deriv);
    n11->dx = backup;

    backup = n11->dy;
    n11->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 14,
        analytic_full_deriv);
    n11->dy = backup;

    backup = n11->dxy;
    n11->dxy += delta ;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_normal_derivative(delta, patch_2, base_normal, 15,
        analytic_full_deriv);
    n11->dxy = backup;
}


TEST(NormalDivergenceTest, Values)
{
    double const patch_to_pixel = 0.2;
    double const patch_to_pixel2 = 0.04;

    double d_00[96];
    BicubicPatch::node_derivatives(0.7, 0.1, d_00, d_00 + 24,
                                   d_00 + 48, d_00 + 72);

    for (int n = 0; n < 4; ++n)
    {
        for (int i = 4; i < 24; ++i)
            d_00[24 * n + i] *= patch_to_pixel;
        for (int i = 12; i < 24; ++i)
            d_00[24 * n + i] *= patch_to_pixel;
    }

    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 10.0; n00->dx = 4.0; n00->dy = 4.0; n00->dxy = -8.0;
    n10->f = 10.0; n10->dx = -4.0; n10->dy = 4.0; n10->dxy = -8.0;
    n01->f = 10.0; n01->dx = 4.0; n01->dy = -4.0; n01->dxy = -8.0;
    n11->f = 10.0; n11->dx = -4.0; n11->dy = -4.0; n11->dxy = -8.0;

    BicubicPatch::Ptr patch_1 = BicubicPatch::create(n00, n10, n01, n11);
    BicubicPatch::Ptr patch_2;

    double p_x = 0.5;
    double p_y = 0.5;
    double w_1 = patch_1->evaluate_f(p_x, p_y);
    double dx_1 = patch_1->evaluate_dx(p_x, p_y) * patch_to_pixel;
    double dy_1 = patch_1->evaluate_dy(p_x, p_y) * patch_to_pixel;
    double dxy_1 = patch_1->evaluate_dxy(p_x, p_y) * patch_to_pixel2;
    double dxx_1 = patch_1->evaluate_dxx(p_x, p_y) * patch_to_pixel2;
    double dyy_1 = patch_1->evaluate_dyy(p_x, p_y) * patch_to_pixel2;


    double x = 100;
    double y = 200;
    double f = 500;
    double inv_f = 1.0 / f;

    double analytic_full_div[6];
    surfderiv::normal_divergence(x, y, f, w_1, dx_1, dy_1,
        dxy_1, dxx_1, dyy_1, analytic_full_div);

    math::Vec3d base_normal;
    surfderiv::fill_normal(x, y, inv_f, w_1, dx_1, dy_1, *base_normal);

    double backup;
    math::Vec3d normal_2;
    double w_2;
    double dx_2;
    double dy_2;
    double delta = 1e-7;

    backup = x;
    x += delta;
    w_2 = patch_1->evaluate_f(p_x + delta * patch_to_pixel, p_y);
    dx_2 = patch_1->evaluate_dx(p_x + delta * patch_to_pixel, p_y)
        * patch_to_pixel;
    dy_2 = patch_1->evaluate_dy(p_x + delta * patch_to_pixel, p_y)
        * patch_to_pixel;
    surfderiv::fill_normal(x, y, inv_f, w_2, dx_2, dy_2, *normal_2);
    double diff_xx = (normal_2[0] - base_normal[0]) / delta;
    double diff_yx = (normal_2[1] - base_normal[1]) / delta;
    double diff_zx = (normal_2[2] - base_normal[2]) / delta;
    x = backup;

    backup = y;
    y += delta;
    w_2 = patch_1->evaluate_f(p_x, p_y + delta * patch_to_pixel);
    dx_2 = patch_1->evaluate_dx(p_x, p_y + delta * patch_to_pixel)
        * patch_to_pixel;
    dy_2 = patch_1->evaluate_dy(p_x, p_y + delta * patch_to_pixel)
        * patch_to_pixel;
    surfderiv::fill_normal(x, y, inv_f, w_2, dx_2, dy_2, *normal_2);
    double diff_yy = (normal_2[1] - base_normal[1]) / delta;
    double diff_xy = (normal_2[0] - base_normal[0]) / delta;
    double diff_zy = (normal_2[2] - base_normal[2]) / delta;
    y = backup;

    EXPECT_NEAR(diff_xx, analytic_full_div[0], 1e-5);
    EXPECT_NEAR(diff_yx, analytic_full_div[1], 1e-5);
    EXPECT_NEAR(diff_zx, analytic_full_div[2], 1e-5);
    EXPECT_NEAR(diff_xy, analytic_full_div[3], 1e-5);
    EXPECT_NEAR(diff_yy, analytic_full_div[4], 1e-5);
    EXPECT_NEAR(diff_zy, analytic_full_div[5], 1e-5);
}


void
compare_divergence_derivative(
    double delta, BicubicPatch::ConstPtr patch_2, int id = 0,
    double * full_base = nullptr, double * full_deriv = nullptr)
{
    double p_x = 0.7;
    double p_y = 0.1;
    double x = 100;
    double y = 200;
    double f = 500;
    double const patch_to_pixel = 0.2;
    double const patch_to_pixel2 = 0.04;

    double w_2 = patch_2->evaluate_f(p_x, p_y);
    double dx_2 = patch_2->evaluate_dx(p_x, p_y) * patch_to_pixel;
    double dy_2 = patch_2->evaluate_dy(p_x, p_y) * patch_to_pixel;
    double dxy_2 = patch_2->evaluate_dxy(p_x, p_y) * patch_to_pixel2;
    double dxx_2 = patch_2->evaluate_dxx(p_x, p_y) * patch_to_pixel2;
    double dyy_2 = patch_2->evaluate_dyy(p_x, p_y) * patch_to_pixel2;

    double full_div[6];
    surfderiv::normal_divergence(x, y, f, w_2, dx_2, dy_2,
        dxy_2, dxx_2, dyy_2, full_div);

    if (full_deriv != nullptr && full_base != nullptr)
        for (int i = 0; i < 6; ++i)
            EXPECT_NEAR(full_deriv[i * 16 + id], (full_div[i] - full_base[i])
                / delta, 1e-5);
}


TEST(NormalDivergenceTest, Derivatives)
{
    double const patch_to_pixel = 0.2;
    double const patch_to_pixel2 = 0.04;

    double d_00[96];
    BicubicPatch::node_derivatives(0.7, 0.1, d_00, d_00 + 24,
        d_00 + 48, d_00 + 72);

    for (int n = 0; n < 4; ++n)
    {
        for (int i = 4; i < 24; ++i)
            d_00[24 * n + i] *= patch_to_pixel;
        for (int i = 12; i < 24; ++i)
            d_00[24 * n + i] *= patch_to_pixel;
    }

    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 10.0; n00->dx = 4.0; n00->dy = 4.0; n00->dxy = -8.0;
    n10->f = 10.0; n10->dx = -4.0; n10->dy = 4.0; n10->dxy = -8.0;
    n01->f = 10.0; n01->dx = 4.0; n01->dy = -4.0; n01->dxy = -8.0;
    n11->f = 10.0; n11->dx = -4.0; n11->dy = -4.0; n11->dxy = -8.0;

    BicubicPatch::Ptr patch_1 = BicubicPatch::create(n00, n10, n01, n11);
    BicubicPatch::Ptr patch_2;

    double p_x = 0.7;
    double p_y = 0.1;
    double w_1 = patch_1->evaluate_f(p_x, p_y);
    double dx_1 = patch_1->evaluate_dx(p_x, p_y) * patch_to_pixel;
    double dy_1 = patch_1->evaluate_dy(p_x, p_y) * patch_to_pixel;
    double dxy_1 = patch_1->evaluate_dxy(p_x, p_y) * patch_to_pixel2;
    double dxx_1 = patch_1->evaluate_dxx(p_x, p_y) * patch_to_pixel2;
    double dyy_1 = patch_1->evaluate_dyy(p_x, p_y) * patch_to_pixel2;


    double x = 100;
    double y = 200;
    double f = 500;

    double full_base_values[6];
    surfderiv::normal_divergence(x, y, f, w_1, dx_1, dy_1,
        dxy_1, dxx_1, dyy_1, full_base_values);

    double full_deriv[96];
    surfderiv::normal_divergence_deriv(d_00, x, y, f, w_1, dx_1, dy_1,
        dxy_1, dxx_1, dyy_1, full_deriv);

    double const delta = 1e-7;
    double backup;

    /* Node 00 */
    backup = n00->f;
    n00->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2,
        0, full_base_values, full_deriv);
    n00->f = backup;

    backup = n00->dx;
    n00->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2,
        1, full_base_values, full_deriv);
    n00->dx = backup;

    backup = n00->dy;
    n00->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2,
        2, full_base_values, full_deriv);
    n00->dy = backup;

    backup = n00->dxy;
    n00->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2,
        3, full_base_values, full_deriv);
    n00->dxy = backup;
#if 1
    /* Node 10 */
    backup = n10->f;
    n10->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2,
        4, full_base_values, full_deriv);
    n10->f = backup;

    backup = n10->dx;
    n10->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2,
        5, full_base_values, full_deriv);
    n10->dx = backup;

    backup = n10->dy;
    n10->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2,
        6, full_base_values, full_deriv);
    n10->dy = backup;

    backup = n10->dxy;
    n10->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2,
        7, full_base_values, full_deriv);
    n10->dxy = backup;

    /* Node 01 */
    backup = n01->f;
    n01->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2);
    n01->f = backup;

    backup = n01->dx;
    n01->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2);
    n01->dx = backup;

    backup = n01->dy;
    n01->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2);
    n01->dy = backup;

    backup = n01->dxy;
    n01->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2);
    n01->dxy = backup;

    /* Node 11 */
    backup = n11->f;
    n11->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2);
    n11->f = backup;

    backup = n11->dx;
    n11->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2);
    n11->dx = backup;

    backup = n11->dy;
    n11->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2);
    n11->dy = backup;

    backup = n11->dxy;
    n11->dxy += delta ;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_divergence_derivative(delta, patch_2,
        15, full_base_values, full_deriv);
    n11->dxy = backup;
#endif
}
