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

#include "spherical_harmonics.h"

TEST(SphericalHarmonicsTest, Derivatives4)
{
    math::Vec3d normal(0.2, 0.3, 0.4);
    normal.normalize();

    math::Vector<double, 16> base_values;
    sh::evaluate_4_band(*normal, *base_values);

    double delta = 1e-7;
    math::Vector<double, 16> test_values;

    double analytic_deriv[48];
    sh::derivative_4_band(*normal, analytic_deriv);

    double backup = normal[0];
    normal[0] += delta;
    sh::evaluate_4_band(*normal, *test_values);
    for(int i = 0; i < 16; ++i)
    {
        double numeric = (test_values[i] - base_values[i]) / delta;
        EXPECT_NEAR(analytic_deriv[i * 3], numeric, 1e-5);
    }
    normal[0] = backup;

    backup = normal[1];
    normal[1] += delta;
    sh::evaluate_4_band(*normal, *test_values);
    for(int i = 0; i < 16; ++i)
    {
        double numeric = (test_values[i] - base_values[i]) / delta;
        EXPECT_NEAR(analytic_deriv[i * 3 + 1], numeric, 1e-5);
    }
    normal[1] = backup;

    backup = normal[2];
    normal[2] += delta;
    sh::evaluate_4_band(*normal, *test_values);
    for(int i = 0; i < 16; ++i)
    {
        double numeric = (test_values[i] - base_values[i]) / delta;
        EXPECT_NEAR(analytic_deriv[i * 3 + 2], numeric, 1e-5);
    }
    normal[2] = backup;
}
