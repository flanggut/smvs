/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "math/matrix.h"
#include "math/matrix_svd.h"

#include "bicubic_patch.h"
#include "ldl_decomposition.h"

SMVS_NAMESPACE_BEGIN

namespace
{
    const double coefficient_matrix[256]
    {
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0,
        -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0,
        9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1,
        -6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1,
        2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0,
        -6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
        4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1
    };

    void
    compute_exponentials (double const x, double const y,
        double *ex, double *ey)
    {
        ex[0] = 1.0;
        ex[1] = x;
        ex[2] = x * x;
        ex[3] = ex[2] * x;

        ey[0] = 1.0;
        ey[1] = y;
        ey[2] = y * y;
        ey[3] = ey[2] * y;
    }
}

void
BicubicPatch::compute_coefficients (void)
{
    math::Matrix<double, 16, 16> A(coefficient_matrix);

    math::Vector<double, 16> x;
    x[0]  = this->n00->f;
    x[1]  = this->n10->f;
    x[2]  = this->n01->f;
    x[3]  = this->n11->f;

    x[4]  = this->n00->dx;
    x[5] = this->n10->dx;
    x[6]  = this->n01->dx;
    x[7] = this->n11->dx;

    x[8]  = this->n00->dy;
    x[9]  = this->n10->dy;
    x[10]  = this->n01->dy;
    x[11]  = this->n11->dy;

    x[12] = this->n00->dxy;
    x[13] = this->n10->dxy;
    x[14] = this->n01->dxy;
    x[15] = this->n11->dxy;

    math::Vector<double, 16> a = A * x;
    for (int k = 0, j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i, ++k)
            this->coeffs[i][j] = a[k];
}

void
BicubicPatch::compute_coefficients_sanity_check (void)
{
    math::Matrix4d L, R;
    L[0]  =  1.0; L[1]  =  0.0; L[2]  =  0.0; L[3]  =  0.0;
    L[4]  =  0.0; L[5]  =  0.0; L[6]  =  1.0; L[7]  =  0.0;
    L[8]  = -3.0; L[9]  =  3.0; L[10] = -2.0; L[11] = -1.0;
    L[12] =  2.0; L[13] = -2.0; L[14] =  1.0; L[15] =  1.0;
    R = L.transposed();

    math::Matrix4d F;
    F[0]  = this->n00->f;
    F[1]  = this->n01->f;
    F[2]  = this->n00->dy;
    F[3]  = this->n01->dy;
    F[4]  = this->n10->f;
    F[5]  = this->n11->f;
    F[6]  = this->n10->dy;
    F[7]  = this->n11->dy;
    F[8]  = this->n00->dx;
    F[9]  = this->n01->dx;
    F[10] = this->n00->dxy;
    F[11] = this->n01->dxy;
    F[12] = this->n10->dx;
    F[13] = this->n11->dx;
    F[14] = this->n10->dxy;
    F[15] = this->n11->dxy;

    math::Matrix4d A;
    A = L * F * R;
    std::copy(A.begin(), A.end(), this->coeffs[0]);
}

double
BicubicPatch::evaluate_f (double const* x, double const* y) const
{
    double result = 0.0f;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            result += this->coeffs[i][j] * x[i] * y[j];

    return result;
}

double
BicubicPatch::evaluate_dx (double const* x, double const* y) const
{
    double result = 0.0f;
    for (int i = 1; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            result += this->coeffs[i][j] * i * x[i-1] * y[j];

    return result;
}

double
BicubicPatch::evaluate_dxx (double const* x, double const* y) const
{
    double result = 0.0f;
    for (int i = 2; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            result += this->coeffs[i][j] * i * (i-1) * x[i-2] * y[j];

    return result;
}


double
BicubicPatch::evaluate_dy (double const* x, double const* y) const
{
    double result = 0.0f;
    for (int i = 0; i < 4; ++i)
        for (int j = 1; j < 4; ++j)
            result += this->coeffs[i][j] * x[i] * j * y[j-1];

    return result;
}

double
BicubicPatch::evaluate_dyy (double const* x, double const* y) const
{
    double result = 0.0f;
    for (int i = 0; i < 4; ++i)
        for (int j = 2; j < 4; ++j)
            result += this->coeffs[i][j] * x[i] * j * (j-1) * y[j-2];

    return result;
}


double
BicubicPatch::evaluate_dxy (double const* x, double const* y) const
{
    double result = 0.0f;
    for (int i = 1; i < 4; ++i)
        for (int j = 1; j < 4; ++j)
            result += this->coeffs[i][j] * i * x[i-1] * j * y[j-1];

    return result;
}

void
BicubicPatch::evaluate_all (double const* x, double const* y,
    double * values) const
{
    values[0] = this->evaluate_f(x, y);
    values[1] = this->evaluate_dx(x, y);
    values[2] = this->evaluate_dy(x, y);
    values[3] = this->evaluate_dxy(x, y);
}


double
BicubicPatch::evaluate_f (double x, double y) const
{
    double ex[4], ey[4];
    compute_exponentials(x, y, ex, ey);
    return this->evaluate_f(ex, ey);
}

double
BicubicPatch::evaluate_dx (double x, double y) const
{
    double ex[4], ey[4];
    compute_exponentials(x, y, ex, ey);
    return this->evaluate_dx(ex, ey);
}

double
BicubicPatch::evaluate_dxx (double x, double y) const
{
    double ex[4], ey[4];
    compute_exponentials(x, y, ex, ey);
    return this->evaluate_dxx(ex, ey);
}


double
BicubicPatch::evaluate_dy (double x, double y) const
{
    double ex[4], ey[4];
    compute_exponentials(x, y, ex, ey);
    return this->evaluate_dy(ex, ey);
}

double
BicubicPatch::evaluate_dyy (double x, double y) const
{
    double ex[4], ey[4];
    compute_exponentials(x, y, ex, ey);
    return this->evaluate_dyy(ex, ey);
}

double
BicubicPatch::evaluate_dxy (double x, double y) const
{
    double ex[4], ey[4];
    compute_exponentials(x, y, ex, ey);
    return this->evaluate_dxy(ex, ey);
}


void
BicubicPatch::evaluate_all (double x, double y, double * values) const
{
    double ex[4], ey[4];
    compute_exponentials(x, y, ex, ey);
    this->evaluate_all(ex, ey, values);
}

void
node_deriv (double const* x, double const* y, int node_offset,
    double * d_f, double * d_dx, double * d_dy, double * d_dxy,
    double * d_dxx = nullptr, double * d_dyy = nullptr)
{
    std::fill_n(d_f, 4, 0.0);
    std::fill_n(d_dx, 4, 0.0);
    std::fill_n(d_dy, 4, 0.0);
    std::fill_n(d_dxy, 4, 0.0);
    if (d_dxx != nullptr)
        std::fill_n(d_dxx, 4, 0.0);
    if (d_dyy != nullptr)
        std::fill_n(d_dyy, 4, 0.0);

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int c = 0; c < 4; ++c)
            {
                int const o = 4 * c + node_offset;
                /* Value derivative */
                d_f[c] +=  coefficient_matrix[16 * (j * 4 + i) + o]
                    * x[i] * y[j];
                /* Dx derivative */
                if(i > 0)
                    d_dx[c] +=  coefficient_matrix[16 * (j * 4 + i) + o]
                        * i * x[i - 1] * y[j];
                /* Dy derivative */
                if(j > 0)
                    d_dy[c] +=  coefficient_matrix[16 * (j * 4 + i) + o]
                        * x[i] * j * y[j - 1];
                /* Mixed second derivative */
                if (i > 0 && j > 0)
                    d_dxy[c] +=  coefficient_matrix[16 * (j * 4 + i) + o]
                        * i * x[i - 1] * j * y[j - 1];
                /* Second derivative */
                if (d_dxx != nullptr && i > 1)
                    d_dxx[c] +=  coefficient_matrix[16 * (j * 4 + i) + o]
                        * i * (i-1) * x[i - 2] * y[j];
                if (d_dyy != nullptr && j > 1)
                    d_dyy[c] +=  coefficient_matrix[16 * (j * 4 + i) + o]
                        * x[i] * j * (j-1) * y[j - 2];
            }
}

void
BicubicPatch::node_derivatives (double x, double y, double * d_00,
    double * d_10, double * d_01, double * d_11)
{
    double ex[4], ey[4];
    compute_exponentials(x, y, ex, ey);
    node_deriv(ex, ey, 0, d_00, d_00 + 4, d_00 + 8,
        d_00 + 12, d_00 + 16, d_00 + 20);
    node_deriv(ex, ey, 1, d_10, d_10 + 4, d_10 + 8,
        d_10 + 12, d_10 + 16, d_10 + 20);
    node_deriv(ex, ey, 2, d_01, d_01 + 4, d_01 + 8,
        d_01 + 12, d_01 + 16, d_01 + 20);
    node_deriv(ex, ey, 3, d_11, d_11 + 4, d_11 + 8,
        d_11 + 12, d_11 + 16, d_11 + 20);
}

void
BicubicPatch::node_derivatives_for_patchsize(double x, double y,
    double patchsize, double *d_00, double *d_10, double *d_01, double *d_11)
{
    node_derivatives(x, y, d_00, d_10, d_01, d_11);
    double patch_to_pixel = 1.0 / patchsize;

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
}

BicubicPatch::Ptr
BicubicPatch::fit_to_data(double const* x, double const* y,
    double const* data, int size)
{
    double ex[4], ey[4];
    std::vector<double> A(size * 16);
    std::vector<double> rhs(size);
    for (int i = 0; i < size; ++i)
    {
        compute_exponentials(x[i], y[i], ex, ey);
        A[i * 16 + 0] = ex[0] * ey[0];
        A[i * 16 + 1] = ex[0] * ey[1];
        A[i * 16 + 2] = ex[0] * ey[2];
        A[i * 16 + 3] = ex[0] * ey[3];
        A[i * 16 + 4] = ex[1] * ey[0];
        A[i * 16 + 5] = ex[1] * ey[1];
        A[i * 16 + 6] = ex[1] * ey[2];
        A[i * 16 + 7] = ex[1] * ey[3];
        A[i * 16 + 8] = ex[2] * ey[0];
        A[i * 16 + 9] = ex[2] * ey[1];
        A[i * 16 + 10] = ex[2] * ey[2];
        A[i * 16 + 11] = ex[2] * ey[3];
        A[i * 16 + 12] = ex[3] * ey[0];
        A[i * 16 + 13] = ex[3] * ey[1];
        A[i * 16 + 14] = ex[3] * ey[2];
        A[i * 16 + 15] = ex[3] * ey[3];

        rhs[i] = data[i];
    }
    math::Matrix<double, 16, 16> AtA;
    std::vector<double> At = A;
    math::matrix_transpose(&At[0], size, 16);
    math::matrix_multiply(&At[0], 16, size, &A[0], 16, &AtA[0]);
    ldl_inverse(&AtA[0], 16);

    std::vector<double> Atrhs(16);
    math::matrix_multiply(&At[0], 16, size, &rhs[0], 1, &Atrhs[0]);

    std::vector<double> alpha(16);
    math::matrix_multiply(&AtA[0], 16, 16, &Atrhs[0], 1, &alpha[0]);

    return Ptr(new BicubicPatch(&alpha[0]));
}

SMVS_NAMESPACE_END
