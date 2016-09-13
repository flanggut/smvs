/*
 * Copyright (C) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <gtest/gtest.h>

#include "bicubic_patch.h"
#include "correspondence.h"

using namespace smvs;

TEST(CorrespondenceTest, Derivatives)
{
    double d_00[24];
    double d_10[24];
    double d_01[24];
    double d_11[24];

    double const u = 0.7;
    double const v = 0.4;
    BicubicPatch::node_derivatives(u, v, d_00, d_10, d_01, d_11);

    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 1.2; n00->dx = 0.2; n00->dy = 0.2; n00->dxy = -0.1;
    n10->f = 1.4; n10->dx = -0.3; n10->dy = 0.3; n10->dxy = -0.2;
    n01->f = 1.1; n01->dx = 0.4; n01->dy = -0.4; n01->dxy = -0.1;
    n11->f = 1.3; n11->dx = -0.2; n11->dy = -0.2; n11->dxy = -0.1;


    BicubicPatch::Ptr patch_1 = BicubicPatch::create(n00, n10, n01, n11);
    BicubicPatch::Ptr patch_2;

    double center_f_1 = patch_1->evaluate_f(u, v);

//    double MM[9] = { 1.90133, 0.0393805, 448.863,
//        -0.12414, 1.98593, 236.985,
//        -3.55536e-05, -7.95936e-06, 1.04958 };
//    double tt[9] = { -6757.74, 2014.51, 0.632036 };

    double MM[9] = {-0.997402, -0.0167178, 626.197,
                    -0.0269324, -1.01116, 174.093,
                    7.05365e-06, -0.000139764, 1.00931};
    double tt[3] = {3.78737, 168.604, 0.0117067};

    math::Matrix3d M(MM);
    math::Vec3d t(tt);


    double const pu  = 100;
    double const pv  = 100;
    Correspondence C(M, t, pu, pv, center_f_1);

    math::Vec2d corr_base;
    C.fill(*corr_base);

    math::Vec2d c_dn00[4];
    math::Vec2d c_dn10[4];
    math::Vec2d c_dn01[4];
    math::Vec2d c_dn11[4];
    C.get_derivative(d_00, d_10, d_01, d_11,
        c_dn00, c_dn10, c_dn01, c_dn11);

    Correspondence C_new;
    math::Vec2d corr_new;
    math::Vec2d diff;

    double delta = 1e-8;
    double backup;
    double center_f_2;

    double const epsilon = 1e-4;
    /* Node 00 */
    backup = n00->f;
    n00->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn00[0][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn00[0][1], diff[1], epsilon);
    n00->f = backup;

    backup = n00->dx;
    n00->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn00[1][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn00[1][1], diff[1], epsilon);
    n00->dx = backup;

    backup = n00->dy;
    n00->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn00[2][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn00[2][1], diff[1], epsilon);
    n00->dy = backup;

    backup = n00->dxy;
    n00->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn00[3][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn00[3][1], diff[1], epsilon);
    n00->dxy = backup;

    /* Node 10 */
    backup = n10->f;
    n10->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn10[0][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn10[0][1], diff[1], epsilon);
    n10->f = backup;

    backup = n10->dx;
    n10->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn10[1][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn10[1][1], diff[1], epsilon);
    n10->dx = backup;

    backup = n10->dy;
    n10->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn10[2][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn10[2][1], diff[1], epsilon);
    n10->dy = backup;

    backup = n10->dxy;
    n10->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn10[3][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn10[3][1], diff[1], epsilon);
    n10->dxy = backup;

    /* Node 01 */
    backup = n01->f;
    n01->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn01[0][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn01[0][1], diff[1], epsilon);
    n01->f = backup;

    backup = n01->dx;
    n01->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn01[1][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn01[1][1], diff[1], epsilon);
    n01->dx = backup;

    backup = n01->dy;
    n01->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn01[2][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn01[2][1], diff[1], epsilon);
    n01->dy = backup;

    backup = n01->dxy;
    n01->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn01[3][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn01[3][1], diff[1], epsilon);
    n01->dxy = backup;

    /* Node 11 */
    backup = n11->f;
    n11->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn11[0][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn11[0][1], diff[1], epsilon);
    n11->f = backup;

    backup = n11->dx;
    n11->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn11[1][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn11[1][1], diff[1], epsilon);
    n11->dx = backup;

    backup = n11->dy;
    n11->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn11[2][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn11[2][1], diff[1], epsilon);
    n11->dy = backup;

    backup = n11->dxy;
    n11->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(u, v);
    C_new = Correspondence(M, t, pu, pv, center_f_2);
    C_new.fill(*corr_new);
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(c_dn11[3][0], diff[0], epsilon);
    EXPECT_NEAR(c_dn11[3][1], diff[1], epsilon);
    n11->dxy = backup;
}


void compare_jacobian_deriv(math::Matrix2d const& jac_base, double delta,
    math::Matrix2d const& jac_analytic, BicubicPatch::Ptr deriv_patch,
    math::Matrix3d const& M, math::Vec3d const& t, double u, double v)
{
    double const patch_to_pixel = 0.8;
    double center_f = deriv_patch->evaluate_f(0.2, 0.8);
    double center_dx = deriv_patch->evaluate_dx(0.2, 0.8) * patch_to_pixel;
    double center_dy = deriv_patch->evaluate_dy(0.2, 0.8) * patch_to_pixel;

    Correspondence const C_new(M, t, u, v, center_f, center_dx, center_dy);
    math::Matrix2d jac_new;
    C_new.fill_jacobian(*jac_new);

    math::Matrix2d diff = (jac_new - jac_base) / delta;

    EXPECT_NEAR(jac_analytic[0], diff[0], 1e-4);
    EXPECT_NEAR(jac_analytic[1], diff[1], 1e-4);
    EXPECT_NEAR(jac_analytic[2], diff[2], 1e-4);
    EXPECT_NEAR(jac_analytic[3], diff[3], 1e-4);
}

void compare_jacobian_deriv_grad(math::Matrix2d const& jac_base, double delta,
    math::Vec2d const& jac_grad_analytic, math::Vec2d const& grad,
    BicubicPatch::Ptr deriv_patch,
    math::Matrix3d const& M, math::Vec3d const& t, double u, double v)
{
    double const patch_to_pixel = 0.8;
    double center_f = deriv_patch->evaluate_f(0.2, 0.8);
    double center_dx = deriv_patch->evaluate_dx(0.2, 0.8) * patch_to_pixel;
    double center_dy = deriv_patch->evaluate_dy(0.2, 0.8) * patch_to_pixel;

    Correspondence const C_new(M, t, u, v, center_f, center_dx, center_dy);
    math::Matrix2d jac_new;
    C_new.fill_jacobian(*jac_new);

    math::Matrix2d diff = (jac_new - jac_base) / delta;
    math::Vec2d diff_grad = diff * grad;

    EXPECT_NEAR(jac_grad_analytic[0], diff_grad[0], 1e-5);
    EXPECT_NEAR(jac_grad_analytic[1], diff_grad[1], 1e-5);
}


TEST(CorrespondenceJacobianTest, ValuesAndDerivatives)
{
    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 8.2; n00->dx = 0.2; n00->dy = 0.2; n00->dxy = 0.0;
    n10->f = 9.4; n10->dx = -0.3; n10->dy = 0.3; n10->dxy = -0.2;
    n01->f = 10.1; n01->dx = 0.4; n01->dy = -0.4; n01->dxy = 0.1;
    n11->f = 3.3; n11->dx = -0.2; n11->dy = -0.2; n11->dxy = -0.1;

    BicubicPatch::Ptr patch_1 = BicubicPatch::create(n00, n10, n01, n11);
    BicubicPatch::Ptr patch_2;

    double const u = 0.2;
    double const v = 0.8;
    double const patch_to_pixel = 0.8;
    double d_n[96];
    BicubicPatch::node_derivatives_for_patchsize(u, v, 1.0 / patch_to_pixel,
         d_n, d_n + 24, d_n + 48, d_n + 72);

    double center_f_1 = patch_1->evaluate_f(u, v);
    double center_dx_1 = patch_1->evaluate_dx(u, v) * patch_to_pixel;
    double center_dy_1 = patch_1->evaluate_dy(u, v) * patch_to_pixel;

    double MM[9] = {-0.997402, -0.0167178, 626.197,
                    -0.0269324, -1.01116, 174.093,
                    7.05365e-06, -0.000139764, 1.00931};
    double tt[3] = {3.78737, 168.604, 0.0117067};
    math::Matrix3d M(MM);
    math::Vec3d t(tt);

    double x = 300 + 0.5;
    double y = 200 + 0.5;

    Correspondence const C(M, t, x, y, center_f_1, center_dx_1, center_dy_1);
    math::Matrix2d jac_base;
    C.fill_jacobian(*jac_base);

    math::Vec2d corr_base;
    C.fill(*corr_base);
    corr_base[0] -= 0.5;
    corr_base[1] -= 0.5;

    math::Matrix2d corr_jac;
    C.fill_jacobian(*corr_jac);

    Correspondence C_new;
    math::Vec2d diff;
    math::Vec2d corr_new;
    double delta = 1e-9;
    double backup;

    backup = x;
    x += delta;
    center_f_1 = patch_1->evaluate_f(u + delta * patch_to_pixel, v);
    C_new = Correspondence(M, t, x, y, center_f_1, center_dx_1, center_dy_1);
    C_new.fill(*corr_new);
    corr_new[0] -= 0.5;
    corr_new[1] -= 0.5;
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(corr_jac[0], diff[0], 1e-4);
    EXPECT_NEAR(corr_jac[1], diff[1], 1e-4);
    x = backup;

    backup = y;
    y += delta;
    center_f_1 = patch_1->evaluate_f(u, v + delta * patch_to_pixel);
    C_new = Correspondence(M, t, x, y, center_f_1, center_dx_1, center_dy_1);
    C_new.fill(*corr_new);
    corr_new[0] -= 0.5;
    corr_new[1] -= 0.5;
    diff = (corr_new - corr_base) / delta;
    EXPECT_NEAR(corr_jac[2], diff[0], 1e-4);
    EXPECT_NEAR(corr_jac[3], diff[1], 1e-4);
    y = backup;

#if 1
    math::Matrix2d j_dn[16];
    C.fill_jacobian_derivative(d_n, j_dn);

    math::Matrix2d * j_dn00 = j_dn;
    math::Matrix2d * j_dn10 = j_dn + 4;
    math::Matrix2d * j_dn01 = j_dn + 8;
    math::Matrix2d * j_dn11 = j_dn + 12;

    math::Vec2d grad(0.2,0.1);
    math::Vec2d j_dn_grad[16];
    C.fill_jacobian_derivative_grad(*grad, d_n, j_dn_grad);

    math::Vec2d * j_dn00_grad = j_dn_grad;
    math::Vec2d * j_dn10_grad = j_dn_grad + 4;
    math::Vec2d * j_dn01_grad = j_dn_grad + 8;
    math::Vec2d * j_dn11_grad = j_dn_grad + 12;

    /* Node 00 */
    backup = n00->f;
    n00->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn00[0], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn00_grad[0], grad,
        patch_2, M, t, x, y);
    n00->f = backup;

    backup = n00->dx;
    n00->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn00[1], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn00_grad[1], grad,
        patch_2, M, t, x, y);
    n00->dx = backup;

    backup = n00->dy;
    n00->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn00[2], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn00_grad[2], grad,
        patch_2, M, t, x, y);
    n00->dy = backup;

    backup = n00->dxy;
    n00->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn00[3], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn00_grad[3], grad,
        patch_2, M, t, x, y);
    n00->dxy = backup;

    /* Node 10 */
    backup = n10->f;
    n10->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn10[0], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn10_grad[0], grad,
        patch_2, M, t, x, y);
    n10->f = backup;

    backup = n10->dx;
    n10->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn10[1], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn10_grad[1], grad,
        patch_2, M, t, x, y);
    n10->dx = backup;

    backup = n10->dy;
    n10->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn10[2], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn10_grad[2], grad,
        patch_2, M, t, x, y);
    n10->dy = backup;

    backup = n10->dxy;
    n10->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn10[3], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn10_grad[3], grad,
        patch_2, M, t, x, y);
    n10->dxy = backup;

    /* Node 01 */
    backup = n01->f;
    n01->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn01[0], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn01_grad[0], grad,
        patch_2, M, t, x, y);
    n01->f = backup;

    backup = n01->dx;
    n01->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn01[1], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn01_grad[1], grad,
        patch_2, M, t, x, y);
    n01->dx = backup;

    backup = n01->dy;
    n01->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn01[2], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn01_grad[2], grad,
                                patch_2, M, t, x, y);
    n01->dy = backup;

    backup = n01->dxy;
    n01->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn01[3], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn01_grad[3], grad,
        patch_2, M, t, x, y);
    n01->dxy = backup;

    /* Node 11 */
    backup = n11->f;
    n11->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn11[0], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn11_grad[0], grad,
        patch_2, M, t, x, y);
    n11->f = backup;

    backup = n11->dx;
    n11->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn11[1], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn11_grad[1], grad,
        patch_2, M, t, x, y);
    n11->dx = backup;

    backup = n11->dy;
    n11->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn11[2], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn11_grad[2], grad,
        patch_2, M, t, x, y);
    n11->dy = backup;

    backup = n11->dxy;
    n11->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    compare_jacobian_deriv(jac_base, delta, j_dn11[3], patch_2, M, t, x, y);
    compare_jacobian_deriv_grad(jac_base, delta, j_dn11_grad[3], grad,
        patch_2, M, t, x, y);
    n11->dxy = backup;
#endif
}
