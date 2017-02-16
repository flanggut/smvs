/*
 * Copyright (c) 2016-2017, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_SPHERICAL_HARMONICS_HEADER
#define SMVS_SPHERICAL_HARMONICS_HEADER

#include "defines.h"

SMVS_NAMESPACE_BEGIN
SPHERICAL_HARMONICS_NAMESPACE_BEGIN

/**
 * spherical harmonics with 9 coefficients for normal vector
 * according to http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
 */
template <typename T>
void
evaluate_3_band_exact(T const* normal, T * sh)
{
    /* l = 0 */
    sh[0] = T(0.28209479177387814347); // 0.5 * sqrt(1/PI)

    /* l = 1 */
    T v1 = T(0.48860251190291992158); // 0.5 * sqrt(3/PI)
    sh[1] = v1 * normal[1];
    sh[2] = v1 * normal[2];
    sh[3] = v1 * normal[0];

    /* l = 2 */
    T v2 = T(.94617469575756001809); // 0.75 * sqrt(5/pi)
    T v3 = T(.31539156525252000603); // 0.25 * sqrt(5/pi)
    sh[6] = v2 * normal[2] * normal[2] - v3;

    T v4 = T(1.09254843059207907054) * normal[2]; // 0.5 * (15/PI) * z
    sh[5] = v4 * normal[1];
    sh[7] = v4 * normal[0];

    sh[4] = T(1.09254843059207907054) * normal[0] * normal[1];
    sh[8] = T(.54627421529603953526) *
        (normal[0] * normal[0] - normal[1] * normal[1]);
}

/**
 * spherical harmonics with 9 coefficients for normal vector
 * rescaled to minimize operations
 */
template <typename T>
void
evaluate_3_band(T const* normal, T * sh)
{
    /* l = 0 */
    sh[0] = T(1.0);

    /* l = 1 */
    sh[1] = normal[1];
    sh[2] = normal[2];
    sh[3] = normal[0];

    /* l = 2 */
    sh[4] = normal[0] * normal[1];
    sh[5] = normal[1] * normal[2];
    sh[6] = -MATH_POW2(normal[0]) - MATH_POW2(normal[1])
        + T(2.0) * MATH_POW2(normal[2]);

    sh[7] = normal[0] * normal[2];
    sh[8] = normal[0] * normal[0] - normal[1] * normal[1];
}

/**
 * derivatives of scaled spherical harmonics with 9 coefficients 
 * for normal vector resulting in a 9x3 Matrix
 */
template <typename T>
void
derivative_3_band(T const* normal, T * sh_deriv)
{
    /* sh 0 */
    sh_deriv[0] = T(0.0);
    sh_deriv[1] = T(0.0);
    sh_deriv[2] = T(0.0);

    /* sh 1 */
    sh_deriv[3] = T(0.0);
    sh_deriv[4] = T(1.0);
    sh_deriv[5] = T(0.0);

    /* sh 2 */
    sh_deriv[6] = T(0.0);
    sh_deriv[7] = T(0.0);
    sh_deriv[8] = T(1.0);

    /* sh 3 */
    sh_deriv[9] = T(1.0);
    sh_deriv[10] = T(0.0);
    sh_deriv[11] = T(0.0);

    /* sh 4 */
    sh_deriv[12] = normal[1];
    sh_deriv[13] = normal[0];
    sh_deriv[14] = T(0.0);

    /* sh 5 */
    sh_deriv[15] = T(0.0);
    sh_deriv[16] = normal[2];
    sh_deriv[17] = normal[1];

    /* sh 6 */
    sh_deriv[18] = T(-2.0) * normal[0];
    sh_deriv[19] = T(-2.0) * normal[1];
    sh_deriv[20] = T( 4.0) * normal[2];

    /* sh 7 */
    sh_deriv[21] = normal[2];
    sh_deriv[22] = T(0.0);
    sh_deriv[23] = normal[0];

    /* sh 8 */
    sh_deriv[24] = T( 2.0) * normal[0];
    sh_deriv[25] = T(-2.0) * normal[1];
    sh_deriv[26] = T(0.0);
}

/**
 * spherical harmonics with 16 coefficients for normal vector
 * rescaled to minimize operations
 */
template <typename T>
void
evaluate_4_band(T const* normal, T * sh)
{
    evaluate_3_band(normal, sh);

    T x2 = MATH_POW2(normal[0]);
    T y2 = MATH_POW2(normal[1]);
    T z2 = MATH_POW2(normal[2]);

    /* l = 3 */
    sh[9] = (T(3.0) * x2 - y2) * normal[1];
    sh[10] = normal[0] * normal[1] * normal[2];
    sh[11] = (T(4.0) * z2 - x2 - y2) * normal[1];
    sh[12] = (T(2.0) * z2 - T(3.0) * x2 - T(3.0) * y2) * normal[2];
    sh[13] = (T(4.0) * z2 - x2 - y2) * normal[0];
    sh[14] = (x2 - y2) * normal[2];
    sh[15] = (x2 - T(3.0) * y2) * normal[0];
}

/**
 * derivatives of scaled spherical harmonics with 16 coefficients
 * for normal vector resulting in a 16x3 Matrix
 */
template <typename T>
void
derivative_4_band(T const* normal, T * sh_deriv)
{
    derivative_3_band(normal, sh_deriv);

    T x2 = MATH_POW2(normal[0]);
    T y2 = MATH_POW2(normal[1]);
    T z2 = MATH_POW2(normal[2]);

    /* sh 9 */
    sh_deriv[27] = T(6.0) * normal[0] * normal[1];
    sh_deriv[28] = T(3.0) * (x2 - y2);
    sh_deriv[29] = T(0.0);

    /* sh 10 */
    sh_deriv[30] = normal[1] * normal[2];
    sh_deriv[31] = normal[0] * normal[2];
    sh_deriv[32] = normal[0] * normal[1];

    /* sh 11 */
    sh_deriv[33] = -T(2.0) * normal[0] * normal[1];
    sh_deriv[34] = T(4.0) * z2 - x2 - T(3.0) * y2;
    sh_deriv[35] = T(8.0) * normal[1] * normal[2];

    /* sh 12 */
    sh_deriv[36] = -T(6.0) * normal[0] * normal[2];
    sh_deriv[37] = -T(6.0) * normal[1] * normal[2];
    sh_deriv[38] = T(6.0) * z2 - T(3.0) * (x2 + y2);

    /* sh 13 */
    sh_deriv[39] = T(4.0) * z2 - T(3.0) * x2 - y2;
    sh_deriv[40] = -T(2.0) * normal[0] * normal[1];
    sh_deriv[41] = T(8.0) * normal[0] * normal[2];

    /* sh 14 */
    sh_deriv[42] = T(2.0) * normal[0] * normal[2];
    sh_deriv[43] = -T(2.0) * normal[1] * normal[2];
    sh_deriv[44] = x2 - y2;

    /* sh 15 */
    sh_deriv[45] = T(3.0) * (x2 - y2);
    sh_deriv[46] = -T(6.0) * normal[0] * normal[1];
    sh_deriv[47] = T(0.0);
}

SPHERICAL_HARMONICS_NAMESPACE_END
SMVS_NAMESPACE_END

#endif /* SMVS_SPHERICAL_HARMONICS_HEADER */
