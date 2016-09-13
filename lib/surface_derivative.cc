/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "math/vector.h"

#include "surface_derivative.h"

SMVS_NAMESPACE_BEGIN
SURFACE_DERIVATIVE_NAMESPACE_BEGIN

void
fill_normal(double x, double y, double inv_flen, double w,
    double dx, double dy, double * n)
{
    math::Vec3d normal;
    normal[0] = dx;
    normal[1] = -dy;
    normal[2] = x * dx + y * dy + w;
    normal[2] *= inv_flen;
    normal.normalize();
    std::copy(normal.begin(), normal.end(), n);
}


void
normal_derivative(double const* d_node, double x, double y, double f,
    double w, double dx, double dy, // double dxy, double dxx, double dyy,
    double * deriv)
{
    double f_sqr_inv = 1.0 / (f*f);
    double a = w + x * dx + y * dy;

    double t = dx*dx + dy*dy + a * a * f_sqr_inv;;
    double n = std::sqrt(t);

    for (int node = 0; node < 4; ++node)
    {
        double const* dn = d_node + 24 * node;
        for (int i = 0; i < 4; ++i)
        {
            double w_prime = dn[0 + i];
            double dx_prime = dn[4 + i];
            double dy_prime = dn[8 + i];
            double a_prime = w_prime + x * dx_prime + y * dy_prime;

            double t_prime_2 = (dx * dx_prime) + (dy * dy_prime) +
                f_sqr_inv * a * a_prime;
            double n_prime = t_prime_2 / n;

            double nx_prime = (dx_prime * n - dx * n_prime) / t;
            double ny_prime = (-dy_prime * n + dy * n_prime) / t;
            double nz_prime = (a_prime * n - a * n_prime) / (t * f);

            deriv[0 + node * 4 + i] = nx_prime;
            deriv[16 + node * 4 + i] = ny_prime;
            deriv[32 + node * 4 + i] = nz_prime;
        }
    }
}



void
normal_divergence (double x, double y, double f, double w,
    double dx, double dy, double dxy, double dxx, double dyy,
    double * div)
{
    double a = (w + x * dx + y * dy);
    double ax = 2.0 * dx + x * dxx + y * dxy;
    double ay = 2.0 * dy + y * dyy + x * dxy;

    double t = a / f;
    t = t * t;
    t += MATH_POW2(dx) + MATH_POW2(dy);
    double n = std::sqrt(t);


    double nx = dx * dxx + dy * dxy;
    nx += (1.0 / (f * f)) * (w + x * dx + y * dy)
        * (dx + dx + x * dxx + y * dxy);
    nx /= n;

    double ny = dx * dxy + dy * dyy;
    ny += (1.0 / (f * f)) * (w + x * dx + y * dy)
        * (dy + dy + x * dxy + y * dyy);
    ny /= n;

    double xx = (dxx * n - dx * nx) / t;
    double yy = (dyy * n - dy * ny) / t;
    double xy = (dxy * n - dx * ny) / t;
    double yx = (dxy * n - dy * nx) / t;
    double zx = (ax * n - a * nx) / (t * f);
    double zy = (ay * n - a * ny) / (t * f);

    div[0] = xx;
    div[1] = -yx;
    div[2] = zx;
    div[3] = xy;
    div[4] = -yy;
    div[5] = zy;
}

void
normal_divergence_deriv (double const* d_node, double x, double y, double f,
    double w, double dx, double dy, double dxy, double dxx, double dyy,
    double * full_deriv)
{
    double f_sqr_inv = 1.0 / (f*f);
    double a = w + x * dx + y * dy;
    double ax = 2.0 * dx + x * dxx + y * dxy;
    double ay = 2.0 * dy + y * dyy + x * dxy;

    double a_f2 = a * f_sqr_inv;
    double t = dx*dx + dy*dy + a * a_f2;
    double n = std::sqrt(t);

    double b = dx * dxx + dy * dxy + a_f2 * (2.0 * dx + x * dxx + y * dxy);
    double c = dx * dxy + dy * dyy + a_f2 * (2.0 * dy + x * dxy + y * dyy);
    double nx = b / n;
    double ny = c / n;

    for (int node = 0; node < 4; ++node)
    {
        double const* dn = d_node + 24 * node;
        for (int i = 0; i < 4; ++i)
        {
            double w_prime = dn[0 + i];
            double dx_prime = dn[4 + i];
            double dy_prime = dn[8 + i];
            double dxy_prime = dn[12 + i];
            double dxx_prime = dn[16 + i];
            double dyy_prime = dn[20 + i];

            double a_prime = w_prime + x * dx_prime + y * dy_prime;
            double ax_prime = 2.0 * dx_prime + x * dxx_prime + y * dxy_prime;
            double ay_prime = 2.0 * dy_prime + y * dyy_prime + x * dxy_prime;

            double t_prime_2 = (dx * dx_prime) + (dy * dy_prime) +
                f_sqr_inv * a * a_prime;
            double n_prime = t_prime_2 / n;

            double b_prime = (dx_prime * dxx + dx * dxx_prime) +
                (dy_prime * dxy + dy * dxy_prime) +
                f_sqr_inv * (a_prime * ax + a * ax_prime);
            double c_prime = (dx_prime * dxy + dx * dxy_prime) +
                (dy_prime * dyy + dy * dyy_prime) +
                f_sqr_inv * (a_prime * ay + a * ay_prime);

            double nx_prime = (b_prime * n - b * n_prime) / t;
            double ny_prime = (c_prime * n - c * n_prime) / t;

            double xx_prime = ((dxx_prime * n + dxx * n_prime
                - dx_prime * nx  - dx * nx_prime) * t
                - (dxx * n - dx * nx) * t_prime_2 * 2.0) / (t*t);
                
            double yy_prime = ((dyy_prime * n + dyy * n_prime
                - dy_prime * ny  - dy * ny_prime) * t
                - (dyy * n - dy * ny) * t_prime_2 * 2.0 ) / (t*t);

            double xy_prime = ((dxy_prime * n + dxy * n_prime
                - dx_prime * ny  - dx * ny_prime) * t
                - (dxy * n - dx * ny) * t_prime_2 * 2.0) / (t*t);

            double yx_prime = ((dxy_prime * n + dxy * n_prime
                - dy_prime * nx  - dy * nx_prime) * t
                - (dxy * n - dy * nx) * t_prime_2 * 2.0) / (t*t);

            double zx_prime = ((ax_prime * n + ax * n_prime
                - a_prime * nx   - a * nx_prime) * t
                - (ax * n - a * nx) * t_prime_2 * 2.0) / (t*t*f);

            double zy_prime = ((ay_prime * n + ay * n_prime
                - a_prime * ny   - a * ny_prime) * t
                - (ay * n - a * ny) * t_prime_2 * 2.0) / (t*t*f);

            full_deriv[0 + node * 4 + i] = xx_prime;
            full_deriv[16 + node * 4 + i] = -yx_prime;
            full_deriv[32 + node * 4 + i] = zx_prime;
            full_deriv[48 + node * 4 + i] = xy_prime;
            full_deriv[64 + node * 4 + i] = -yy_prime;
            full_deriv[80 + node * 4 + i] = zy_prime;
        }
    }
}


double
mean_curvature (double dx, double dy, double dxy, double dxx, double dyy)
{
    double dx2 = dx*dx;
    double dy2 = dy*dy;
    double c = (1.0 + dx2) * dyy - 2.0 * dx * dy * dxy + (1.0 + dy2) * dxx;
    double denom = 1.0 + dx2 + dy2;
    denom = std::sqrt(denom * denom * denom);
    c /= denom;
    return c;
}

void
mean_curvature_derivative (double const* d_node, double dx,
    double dy, double dxy, double dxx, double dyy, double * deriv)
{
    double dx2 = dx * dx;
    double dy2 = dy * dy;
    double u = dyy + dx2 * dyy - 2.0 * dx * dy * dxy + dxx + dy2 * dxx;
    double v = std::sqrt(1 + dx2 + dy2);
    double backup_v_prime = 1.5 * v;
    v = v * v * v;

    for (int n = 0; n < 4; ++n)
    {
        double const* dn = d_node + 24 * n;
        for (int i = 0; i < 4; ++i)
        {
//            double dx_prime = dn[4 + i];
//            double dy_prime = dn[8 + i];
//            double dxy_prime = dn[12 + i];
//            double dxx_prime = dn[16 + i];
//            double dyy_prime = dn[20 + i];
            double v_prime = backup_v_prime * 2.0 * (dx * dn[4 + i]
                + dy * dn[8 + i]);
            double u_prime = dn[20 + i] + 2.0 * dx * dn[4 + i] * dyy
                + dx2 * dn[20 + i];
            u_prime -= 2.0 * ((dn[4 + i] * dy + dx * dn[8 + i]) * dxy
                + dx * dy * dn[12 + i]);
            u_prime += dn[16 + i] + 2.0 * dy * dn[8 + i] * dxx
                + dy2 * dn[16 + i];

            deriv[n * 4 + i] = (u_prime * v - u * v_prime) / (v*v);
        }
    }
}

SURFACE_DERIVATIVE_NAMESPACE_END
SMVS_NAMESPACE_END
