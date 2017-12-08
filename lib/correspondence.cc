/*
 * Copyright (c) 2017, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "correspondence.h"

SMVS_NAMESPACE_BEGIN

Correspondence::Correspondence (math::Matrix3d const& M, math::Vec3d const& t,
    double u, double v, double w, double w_dx, double w_dy)
{
    this->update(M, t, u, v, w, w_dx, w_dy);
}

void
Correspondence::update (math::Matrix3d const& M, math::Vec3d const& t,
    double u, double v, double w, double w_dx, double w_dy)
{
    this->t = t;
    this->w = w;
    this->p_prime[0] = M(0,0);
    this->p_prime[1] = M(0,1);
    this->q_prime[0] = M(1,0);
    this->q_prime[1] = M(1,1);
    this->r_prime[0] = M(2,0);
    this->r_prime[1] = M(2,1);

    this->w_prime[0] = w_dx;
    this->w_prime[1] = w_dy;

    this->p = M(0,0) * u + M(0,1) * v + M(0,2);
    this->q = M(1,0) * u + M(1,1) * v + M(1,2);
    this->r = M(2,0) * u + M(2,1) * v + M(2,2);

    this->a = w * p + t[0];
    this->b = w * q + t[1];
    this->d = w * r + t[2];
    this->d2 = d * d;
}

void
Correspondence::fill (double * corr) const
{
    corr[0] = a / d;
    corr[1] = b / d;
}

void
Correspondence::get_derivative (
    double const* dn00, double const* dn10,
    double const* dn01, double const* dn11,
    math::Vec2d * c_dn00,
    math::Vec2d * c_dn10,
    math::Vec2d * c_dn01,
    math::Vec2d * c_dn11) const
{
    double du_w = (p * d - r * a) / d2;
    double dv_w = (q * d - r * b) / d2;

    for (int i = 0; i < 4; ++i)
    {
        c_dn00[i] = math::Vec2d(du_w, dv_w) * dn00[i];
        c_dn10[i] = math::Vec2d(du_w, dv_w) * dn10[i];
        c_dn01[i] = math::Vec2d(du_w, dv_w) * dn01[i];
        c_dn11[i] = math::Vec2d(du_w, dv_w) * dn11[i];
    }
}

void
Correspondence::fill_derivative (double const* dn, math::Vec2d * c_dn) const
{
    double du_w = (p * d - r * a) / d2;
    double dv_w = (q * d - r * b) / d2;

    for (int n = 0; n < 4; ++n)
    for (int i = 0; i < 4; ++i)
    {
        c_dn[n * 4 + i][0] = du_w * dn[n * 24 + i];
        c_dn[n * 4 + i][1] = dv_w * dn[n * 24 + i];
    }
}

void
Correspondence::fill_jacobian(double * jac) const
{
    jac[0] = (w_prime[0] * p  + w * p_prime[0]) / d;
    jac[2] = (w_prime[1] * p  + w * p_prime[1]) / d;
    jac[0] -= a * (w_prime[0] * r + w * r_prime[0]) / d2;
    jac[2] -= a * (w_prime[1] * r + w * r_prime[1]) / d2;

    jac[1] = (w_prime[0] * q  + w * q_prime[0]) / d;
    jac[3] = (w_prime[1] * q  + w * q_prime[1]) / d;
    jac[1] -= b * (w_prime[0] * r + w * r_prime[0]) / d2;
    jac[3] -= b * (w_prime[1] * r + w * r_prime[1]) / d2;
}

void
Correspondence::fill_jacobian_derivative_grad(double const* grad,
    double const* dn, math::Vec2d * jac_dn) const
{
    double d4 = d2 * d2;
    double d_prime = 2.0 * d * r;

    math::Vec2d du_a_temp;
    du_a_temp[0] = w * (p_prime[0] * r - p * r_prime[0]);
    du_a_temp[1] = w * (p_prime[1] * r - p * r_prime[1]);
    math::Vec2d du_a_prime;
    du_a_prime[0] = 2.0 * du_a_temp[0];
    du_a_prime[1] = 2.0 * du_a_temp[1];

    math::Vec2d du_b_prime;
    du_b_prime[0] = (p_prime[0] * t[2] - r_prime[0] * t[0]);
    du_b_prime[1] = (p_prime[1] * t[2] - r_prime[1] * t[0]);

    double du_c_prime = (p * t[2] - r * t[0]);
    math::Vec2d du_c;
    du_c[0] = w_prime[0] * du_c_prime;
    du_c[1] = w_prime[1] * du_c_prime;

    math::Vec2d dv_a_temp;
    dv_a_temp[0] = w * (q_prime[0] * r - q * r_prime[0]);
    dv_a_temp[1] = w * (q_prime[1] * r - q * r_prime[1]);
    math::Vec2d dv_a_prime;
    dv_a_prime[0] = 2.0 * dv_a_temp[0];
    dv_a_prime[1] = 2.0 * dv_a_temp[1];

    math::Vec2d dv_b_prime;
    dv_b_prime[0] = (q_prime[0] * t[2] - r_prime[0] * t[1]);
    dv_b_prime[1] = (q_prime[1] * t[2] - r_prime[1] * t[1]);

    double dv_c_prime = (q * t[2] - r * t[1]);
    math::Vec2d dv_c;
    dv_c[0] = w_prime[0] * dv_c_prime;
    dv_c[1] = w_prime[1] * dv_c_prime;

    math::Vec2d du_a_b_c;
    du_a_b_c[0] = w * (du_a_temp[0] + du_b_prime[0]) + du_c[0];
    du_a_b_c[1] = w * (du_a_temp[1] + du_b_prime[1]) + du_c[1];
    math::Vec2d dv_a_b_c;
    dv_a_b_c[0] = w * (dv_a_temp[0] + dv_b_prime[0]) + dv_c[0];
    dv_a_b_c[1] = w * (dv_a_temp[1] + dv_b_prime[1]) + dv_c[1];

    math::Vec2d du_ap_bp_d;
    du_ap_bp_d[0] = (du_a_prime[0] + du_b_prime[0]) / d2;
    du_ap_bp_d[1] = (du_a_prime[1] + du_b_prime[1]) / d2;
    math::Vec2d dv_ap_bp_d;
    dv_ap_bp_d[0] = (dv_a_prime[0] + dv_b_prime[0]) / d2;
    dv_ap_bp_d[1] = (dv_a_prime[1] + dv_b_prime[1]) / d2;
    math::Vec2d du_a_b_c_d_prime;
    du_a_b_c_d_prime[0] = du_a_b_c[0] * d_prime / d4;
    du_a_b_c_d_prime[1] = du_a_b_c[1] * d_prime / d4;
    math::Vec2d dv_a_b_c_d_prime;
    dv_a_b_c_d_prime[0] = dv_a_b_c[0] * d_prime / d4;
    dv_a_b_c_d_prime[1] = dv_a_b_c[1] * d_prime / d4;
    double du_c_prime_d = du_c_prime / d2;
    double dv_c_prime_d = dv_c_prime / d2;

    math::Vec2d du_ap_bp_d_du_a_b_c_d_prime;
    du_ap_bp_d_du_a_b_c_d_prime[0] = du_ap_bp_d[0] - du_a_b_c_d_prime[0];
    du_ap_bp_d_du_a_b_c_d_prime[1] = du_ap_bp_d[1] - du_a_b_c_d_prime[1];
    math::Vec2d dv_ap_bp_d_dv_a_b_c_d_prime;
    dv_ap_bp_d_dv_a_b_c_d_prime[0] = dv_ap_bp_d[0] - dv_a_b_c_d_prime[0];
    dv_ap_bp_d_dv_a_b_c_d_prime[1] = dv_ap_bp_d[1] - dv_a_b_c_d_prime[1];

    math::Vec2d du_dn;
    math::Vec2d dv_dn;
    for (int n = 0; n < 4; ++n)
        for (int i = 0; i < 4; ++i)
        {
            int offset = n * 24;
            du_dn[0] = du_ap_bp_d_du_a_b_c_d_prime[0] * dn[offset + 0 + i];
            du_dn[1] = du_ap_bp_d_du_a_b_c_d_prime[1] * dn[offset + 0 + i];
            dv_dn[0] = dv_ap_bp_d_dv_a_b_c_d_prime[0] * dn[offset + 0 + i];
            dv_dn[1] = dv_ap_bp_d_dv_a_b_c_d_prime[1] * dn[offset + 0 + i];
            jac_dn[n * 4 + i][0] =
                (du_dn[0] + du_c_prime_d * dn[offset + 4 + i]) * grad[0]
                + (dv_dn[0] + dv_c_prime_d * dn[offset + 4 + i]) * grad[1];
            jac_dn[n * 4 + i][1] =
                (du_dn[1] + du_c_prime_d * dn[offset + 8 + i]) * grad[0]
                + (dv_dn[1] + dv_c_prime_d * dn[offset + 8 + i]) * grad[1];
        }
}

SMVS_NAMESPACE_END
