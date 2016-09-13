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

using namespace smvs;

TEST(BicubicPatchTest, LinearX)
{
    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 0.0;
    n00->dx = 1.0;
    n00->dy = 0.0;
    n00->dxy = 0.0;

    n10->f = 1.0;
    n10->dx = 1.0;
    n10->dy = 0.0;
    n10->dxy = 0.0;

    n01->f = 0.0;
    n01->dx = 1.0;
    n01->dy = 0.0;
    n01->dxy = 0.0;

    n11->f = 1.0;
    n11->dx = 1.0;
    n11->dy = 0.0;
    n11->dxy = 0.0;

    BicubicPatch::Ptr patch = BicubicPatch::create(n00, n10, n01, n11);
    EXPECT_NEAR(0.5, patch->evaluate_f(0.5, 0.5), 1e-20);
    EXPECT_NEAR(1.0, patch->evaluate_dx(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dy(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dxy(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dxx(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dyy(0.5, 0.5), 1e-20);
}

TEST(BicubicPatchTest, LinearY)
{
    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 0.0;
    n00->dx = 0.0;
    n00->dy = 1.0;
    n00->dxy = 0.0;

    n10->f = 0.0;
    n10->dx = 0.0;
    n10->dy = 1.0;
    n10->dxy = 0.0;

    n01->f = 1.0;
    n01->dx = 0.0;
    n01->dy = 1.0;
    n01->dxy = 0.0;

    n11->f = 1.0;
    n11->dx = 0.0;
    n11->dy = 1.0;
    n11->dxy = 0.0;

    BicubicPatch::Ptr patch = BicubicPatch::create(n00, n10, n01, n11);
    EXPECT_NEAR(0.5, patch->evaluate_f(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dx(0.5, 0.5), 1e-20);
    EXPECT_NEAR(1.0, patch->evaluate_dy(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dxy(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dxx(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dyy(0.5, 0.5), 1e-20);
}

TEST(BicubicPatchTest, LinearXY)
{
    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 0.0;
    n00->dx = 0.5;
    n00->dy = 0.5;
    n00->dxy = 0.0;

    n10->f = 0.5;
    n10->dx = 0.5;
    n10->dy = 0.5;
    n10->dxy = 0.0;

    n01->f = 0.5;
    n01->dx = 0.5;
    n01->dy = 0.5;
    n01->dxy = 0.0;

    n11->f = 1.0;
    n11->dx = 0.5;
    n11->dy = 0.5;
    n11->dxy = 0.0;

    BicubicPatch::Ptr patch = BicubicPatch::create(n00, n10, n01, n11);
    EXPECT_NEAR(0.5, patch->evaluate_f(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.5, patch->evaluate_dx(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.5, patch->evaluate_dy(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dxy(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dxx(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dyy(0.5, 0.5), 1e-20);
}

TEST(BicubicPatchTest, Quadratic)
{
    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 10.0;
    n00->dx = 4.0;
    n00->dy = 4.0;
    n00->dxy = -8.0;

    n10->f = 10.0;
    n10->dx = -4.0;
    n10->dy = 4.0;
    n10->dxy = -8.0;

    n01->f = 10.0;
    n01->dx = 4.0;
    n01->dy = -4.0;
    n01->dxy = -8.0;

    n11->f = 10.0;
    n11->dx = -4.0;
    n11->dy = -4.0;
    n11->dxy = -8.0;

    BicubicPatch::Ptr patch = BicubicPatch::create(n00, n10, n01, n11);
    EXPECT_NEAR(11.0, patch->evaluate_f(0.5, 0.0), 1e-20);
    EXPECT_NEAR(11.0, patch->evaluate_f(0.0, 0.5), 1e-20);
    EXPECT_NEAR(12.0, patch->evaluate_f(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dx(0.5, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dx(0.5, 0.0), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dy(0.0, 0.5), 1e-20);
    EXPECT_NEAR(0.0, patch->evaluate_dy(0.5, 0.5), 1e-20);
    EXPECT_NEAR(-2.0, patch->evaluate_dxy(0.5, 0.5), 1e-20);
    EXPECT_NEAR(-8.0, patch->evaluate_dxx(0.5, 0.5), 1e-20);
    EXPECT_NEAR(-8.0, patch->evaluate_dyy(0.5, 0.5), 1e-20);
}

TEST(BicubicPatchTest, Derivatives)
{
    BicubicPatch::Node::Ptr n00 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n10 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n01 = BicubicPatch::Node::create();
    BicubicPatch::Node::Ptr n11 = BicubicPatch::Node::create();

    n00->f = 1.0;
    n00->dx = 2.0;
    n00->dy = 2.0;
    n00->dxy = -4.0;

    n10->f = 1.0;
    n10->dx = -2.0;
    n10->dy = 2.0;
    n10->dxy = -4.0;

    n01->f = 1.0;
    n01->dx = 2.0;
    n01->dy = -2.0;
    n01->dxy = -4.0;

    n11->f = 1.0;
    n11->dx = -2.0;
    n11->dy = -2.0;
    n11->dxy = -4.0;

    BicubicPatch::Ptr patch_1 = BicubicPatch::create(n00, n10, n01, n11);
    BicubicPatch::Ptr patch_2;

    double const coord_x = 0.9;
    double const coord_y = 0.3;

    double patch_to_pixel = 0.2;
    double pp2 = 0.04;
    double center_f_1 = patch_1->evaluate_f(coord_x, coord_y);
    double center_dx_1 = patch_1->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    double center_dy_1 = patch_1->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    double center_dxy_1 = patch_1->evaluate_dxy(coord_x, coord_y) * pp2;
    double center_dxx_1 = patch_1->evaluate_dxx(coord_x, coord_y) * pp2;
    double center_dyy_1 = patch_1->evaluate_dyy(coord_x, coord_y) * pp2;

    double d_00[24];
    double d_10[24];
    double d_01[24];
    double d_11[24];
    BicubicPatch::node_derivatives(coord_x, coord_y, d_00, d_10, d_01, d_11);


    for (int i = 4; i < 24; ++i)
    {
        d_00[i] *= patch_to_pixel;
        d_10[i] *= patch_to_pixel;
        d_01[i] *= patch_to_pixel;
        d_11[i] *= patch_to_pixel;
    }
    for (int i = 12; i < 24; ++i)
    {
        d_00[i] *= patch_to_pixel;
        d_10[i] *= patch_to_pixel;
        d_01[i] *= patch_to_pixel;
        d_11[i] *= patch_to_pixel;
    }


    double backup;
    double diff;
    double center_f_2;
    double center_dx_2;
    double center_dy_2;
    double center_dxy_2;
    double center_dxx_2;
    double center_dyy_2;

    double const delta = 1e-4;
    double const near = 1e-8;

    /* Node 00 */
    backup = n00->f;
    n00->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_00[0], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_00[4], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_00[8], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_00[12], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_00[16], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_00[20], near);
    n00->f = backup;

    backup = n00->dx;
    n00->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_00[1], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_00[5], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_00[9], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_00[13], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_00[17], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_00[21], near);
    n00->dx = backup;

#if 1
    backup = n00->dy;
    n00->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_00[2], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_00[6], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_00[10], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_00[14], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_00[18], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_00[22], near);
    n00->dy = backup;

    backup = n00->dxy;
    n00->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_00[3], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_00[7], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_00[11], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_00[15], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_00[19], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_00[23], near);
    n00->dxy = backup;

    /* Node 10 */
    backup = n10->f;
    n10->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_10[0], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_10[4], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_10[8], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_10[12], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_10[16], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_10[20], near);
    n10->f = backup;

    backup = n10->dx;
    n10->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_10[1], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_10[5], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_10[9], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_10[13], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_10[17], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_10[21], near);
    n10->dx = backup;

    backup = n10->dy;
    n10->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_10[2], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_10[6], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_10[10], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_10[14], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_10[18], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_10[22], near);
    n10->dy = backup;

    backup = n10->dxy;
    n10->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_10[3], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_10[7], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_10[11], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_10[15], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_10[19], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_10[23], near);
    n10->dxy = backup;

    /* Node 01 */
    backup = n01->f;
    n01->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_01[0], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_01[4], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_01[8], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_01[12], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_01[16], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_01[20], near);
    n01->f = backup;

    backup = n01->dx;
    n01->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_01[1], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_01[5], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_01[9], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_01[13], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_01[17], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_01[21], near);
    n01->dx = backup;

    backup = n01->dy;
    n01->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_01[2], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_01[6], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_01[10], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_01[14], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_01[18], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_01[22], near);
    n01->dy = backup;

    backup = n01->dxy;
    n01->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_01[3], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_01[7], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_01[11], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_01[15], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_01[19], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_01[23], near);
    n01->dxy = backup;

    /* Node 11 */
    backup = n11->f;
    n11->f += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_11[0], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_11[4], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_11[8], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_11[12], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_11[16], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_11[20], near);
    n11->f = backup;

    backup = n11->dx;
    n11->dx += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_11[1], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_11[5], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_11[9], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_11[13], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_11[17], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_11[21], near);
    n11->dx = backup;

    backup = n11->dy;
    n11->dy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_11[2], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_11[6], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_11[10], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_11[14], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_11[18], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_11[22], near);
    n11->dy = backup;

    backup = n11->dxy;
    n11->dxy += delta;
    patch_2 = BicubicPatch::create(n00, n10, n01, n11);
    center_f_2 = patch_2->evaluate_f(coord_x, coord_y);
    center_dx_2 = patch_2->evaluate_dx(coord_x, coord_y) * patch_to_pixel;
    center_dy_2 = patch_2->evaluate_dy(coord_x, coord_y) * patch_to_pixel;
    center_dxy_2 = patch_2->evaluate_dxy(coord_x, coord_y) * pp2;
    center_dxx_2 = patch_2->evaluate_dxx(coord_x, coord_y) * pp2;
    center_dyy_2 = patch_2->evaluate_dyy(coord_x, coord_y) * pp2;
    diff = (center_f_2 - center_f_1) / delta;
    EXPECT_NEAR(diff, d_11[3], near);
    diff = (center_dx_2 - center_dx_1) / delta;
    EXPECT_NEAR(diff, d_11[7], near);
    diff = (center_dy_2 - center_dy_1) / delta;
    EXPECT_NEAR(diff, d_11[11], near);
    diff = (center_dxy_2 - center_dxy_1) / delta;
    EXPECT_NEAR(diff, d_11[15], near);
    diff = (center_dxx_2 - center_dxx_1) / delta;
    EXPECT_NEAR(diff, d_11[19], near);
    diff = (center_dyy_2 - center_dyy_1) / delta;
    EXPECT_NEAR(diff, d_11[23], near);
    n11->dxy = backup;
#endif
}


TEST(BicubicPatchTest, FitToDataConst)
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> data;

    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
        {
            x.push_back(0.1 * i);
            y.push_back(0.1 * j);
            data.push_back(0.5);
        }

    BicubicPatch::Ptr patch = BicubicPatch::fit_to_data(x.data(), y.data(),
        data.data(), data.size());

    EXPECT_NEAR(0.5, patch->evaluate_f(0.5, 0.5), 1e-5);
    EXPECT_NEAR(0.0, patch->evaluate_dx(0.5, 0.5), 1e-5);
    EXPECT_NEAR(0.0, patch->evaluate_dy(0.5, 0.5), 1e-5);
    EXPECT_NEAR(0.0, patch->evaluate_dxy(0.5, 0.5), 1e-5);
}

TEST(BicubicPatchTest, FitToDataLinearX)
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> data;

    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
        {
            x.push_back(0.1 * i);
            y.push_back(0.1 * j);
            data.push_back(0.5 + 0.01 * i);
        }

    BicubicPatch::Ptr patch = BicubicPatch::fit_to_data(x.data(), y.data(),
        data.data(), data.size());

    EXPECT_NEAR(0.55, patch->evaluate_f(0.5, 0.2), 1e-5);
    EXPECT_NEAR(0.55, patch->evaluate_f(0.5, 0.5), 1e-5);
    EXPECT_NEAR(0.55, patch->evaluate_f(0.5, 0.7), 1e-5);

    EXPECT_NEAR(0.1, patch->evaluate_dx(0.5, 0.5), 1e-5);
    EXPECT_NEAR(0.0, patch->evaluate_dy(0.5, 0.5), 1e-5);
}

TEST(BicubicPatchTest, FitToDataLinearY)
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> data;

    for (int i = 0; i < 10; ++i)
        for (int j = 0; j < 10; ++j)
        {
            x.push_back(0.1 * i);
            y.push_back(0.1 * j);
            data.push_back(0.5 + 0.01 * j);
        }

    BicubicPatch::Ptr patch = BicubicPatch::fit_to_data(x.data(), y.data(),
        data.data(), data.size());

    EXPECT_NEAR(0.55, patch->evaluate_f(0.2, 0.5), 1e-5);
    EXPECT_NEAR(0.55, patch->evaluate_f(0.5, 0.5), 1e-5);
    EXPECT_NEAR(0.55, patch->evaluate_f(0.7, 0.5), 1e-5);

    EXPECT_NEAR(0.0, patch->evaluate_dx(0.5, 0.5), 1e-5);
    EXPECT_NEAR(0.1, patch->evaluate_dy(0.5, 0.5), 1e-5);
}
