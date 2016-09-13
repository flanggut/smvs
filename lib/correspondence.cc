/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "correspondence.h"

SMVS_NAMESPACE_BEGIN

Correspondence::Correspondence(math::Matrix3d const& M, math::Vec3d const& t,
    double u, double v, double w, double w_dx, double w_dy)
    : t(t)
{
    this->w = w;
    this->p_prime[0] = M(0,0);
    this->p_prime[1] = M(0,1);
    this->q_prime[0] = M(1,0);
    this->q_prime[1] = M(1,1);
    this->r_prime[0] = M(2,0);
    this->r_prime[1] = M(2,1);

    this->w_prime = math::Vec2d(w_dx, w_dy);

    math::Vec3d pqr = M * math::Vec3d(u, v, 1.0f);
    this->p = pqr[0];
    this->q = pqr[1];
    this->r = pqr[2];

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
    math::Vec2d du;
    du = (w_prime * p  + w * p_prime) / d;
    du -= a * (w_prime * r + w * r_prime) / d2;

    math::Vec2d dv;
    dv = (w_prime * q  + w * q_prime) / d;
    dv -= b * (w_prime * r + w * r_prime) / d2;

    jac[0] = du[0];
    jac[2] = du[1];
    jac[1] = dv[0];
    jac[3] = dv[1];
}

void
Correspondence::fill_jacobian_derivative(
    double const* dn00, double const* dn10,
    double const* dn01, double const* dn11,
    math::Matrix2d * jac_dn00,
    math::Matrix2d * jac_dn10,
    math::Matrix2d * jac_dn01,
    math::Matrix2d * jac_dn11) const
{
    double w2 = w * w;
    double d4 = d2 * d2;
    double d_prime = 2.0 * d * r;

    math::Vec2d du_a_temp = p_prime * r - p * r_prime;
    math::Vec2d du_a = w2 * du_a_temp;
    math::Vec2d du_a_prime = 2.0 * w * du_a_temp;

    math::Vec2d du_b_prime = (p_prime * t[2] - r_prime * t[0]);
    math::Vec2d du_b = w * du_b_prime;

    double du_c_prime = (p * t[2] - r * t[0]);
    math::Vec2d du_c = w_prime * du_c_prime;

    math::Vec2d dv_a_temp = q_prime * r - q * r_prime;
    math::Vec2d dv_a = w2 * dv_a_temp;
    math::Vec2d dv_a_prime = 2.0 * w * dv_a_temp;

    math::Vec2d dv_b_prime = (q_prime * t[2] - r_prime * t[1]);
    math::Vec2d dv_b = w * dv_b_prime;

    double dv_c_prime = (q * t[2] - r * t[1]);
    math::Vec2d dv_c = w_prime * dv_c_prime;

    math::Vec2d du_a_b_c = du_a + du_b + du_c;
    math::Vec2d dv_a_b_c = dv_a + dv_b + dv_c;

    math::Vec2d du_ap_bp_d = (du_a_prime + du_b_prime) / d2;
    math::Vec2d dv_ap_bp_d = (dv_a_prime + dv_b_prime) / d2;
    math::Vec2d du_a_b_c_d_prime = du_a_b_c * d_prime / d4;
    math::Vec2d dv_a_b_c_d_prime = dv_a_b_c * d_prime / d4;
    double du_c_prime_d = du_c_prime / d2;
    double dv_c_prime_d = dv_c_prime / d2;

    math::Vec2d du_ap_bp_d_du_a_b_c_d_prime = du_ap_bp_d - du_a_b_c_d_prime;
    math::Vec2d dv_ap_bp_d_dv_a_b_c_d_prime = dv_ap_bp_d - dv_a_b_c_d_prime;

    math::Vec2d du_dn;
    math::Vec2d dv_dn;

    /* n00 */
    for (int i = 0; i < 4; ++i)
    {
        du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn00[0 + i];
        dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn00[0 + i];
        jac_dn00[i][0] = du_dn[0] + du_c_prime_d * dn00[4 + i];
        jac_dn00[i][2] = du_dn[1] + du_c_prime_d * dn00[8 + i];
        jac_dn00[i][1] = dv_dn[0] + dv_c_prime_d * dn00[4 + i];
        jac_dn00[i][3] = dv_dn[1] + dv_c_prime_d * dn00[8 + i];
    }
    /* n10 */
    for (int i = 0; i < 4; ++i)
    {
        du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn10[0 + i];
        dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn10[0 + i];
        jac_dn10[i][0] = du_dn[0] + du_c_prime_d * dn10[4 + i];
        jac_dn10[i][2] = du_dn[1] + du_c_prime_d * dn10[8 + i];
        jac_dn10[i][1] = dv_dn[0] + dv_c_prime_d * dn10[4 + i];
        jac_dn10[i][3] = dv_dn[1] + dv_c_prime_d * dn10[8 + i];
    }
    /* n01 */
    for (int i = 0; i < 4; ++i)
    {
        du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn01[0 + i];
        dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn01[0 + i];
        jac_dn01[i][0] = du_dn[0] + du_c_prime_d * dn01[4 + i];
        jac_dn01[i][2] = du_dn[1] + du_c_prime_d * dn01[8 + i];
        jac_dn01[i][1] = dv_dn[0] + dv_c_prime_d * dn01[4 + i];
        jac_dn01[i][3] = dv_dn[1] + dv_c_prime_d * dn01[8 + i];
    }
    /* n11 */
    for (int i = 0; i < 4; ++i)
    {
        du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn11[0 + i];
        dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn11[0 + i];
        jac_dn11[i][0] = du_dn[0] + du_c_prime_d * dn11[4 + i];
        jac_dn11[i][2] = du_dn[1] + du_c_prime_d * dn11[8 + i];
        jac_dn11[i][1] = dv_dn[0] + dv_c_prime_d * dn11[4 + i];
        jac_dn11[i][3] = dv_dn[1] + dv_c_prime_d * dn11[8 + i];
    }
}

void
Correspondence::fill_jacobian_derivative_grad(
    math::Vec2d const& grad,
    double const* dn00, double const* dn10,
    double const* dn01, double const* dn11,
    math::Vec2d * jac_dn00,
    math::Vec2d * jac_dn10,
    math::Vec2d * jac_dn01,
    math::Vec2d * jac_dn11) const
{
    double w2 = w * w;
    double d4 = d2 * d2;
    double d_prime = 2.0 * d * r;

    math::Vec2d du_a_temp = p_prime * r - p * r_prime;
    math::Vec2d du_a = w2 * du_a_temp;
    math::Vec2d du_a_prime = 2.0 * w * du_a_temp;

    math::Vec2d du_b_prime = (p_prime * t[2] - r_prime * t[0]);
    math::Vec2d du_b = w * du_b_prime;

    double du_c_prime = (p * t[2] - r * t[0]);
    math::Vec2d du_c = w_prime * du_c_prime;

    math::Vec2d dv_a_temp = q_prime * r - q * r_prime;
    math::Vec2d dv_a = w2 * dv_a_temp;
    math::Vec2d dv_a_prime = 2.0 * w * dv_a_temp;

    math::Vec2d dv_b_prime = (q_prime * t[2] - r_prime * t[1]);
    math::Vec2d dv_b = w * dv_b_prime;

    double dv_c_prime = (q * t[2] - r * t[1]);
    math::Vec2d dv_c = w_prime * dv_c_prime;

    math::Vec2d du_a_b_c = du_a + du_b + du_c;
    math::Vec2d dv_a_b_c = dv_a + dv_b + dv_c;

    math::Vec2d du_ap_bp_d = (du_a_prime + du_b_prime) / d2;
    math::Vec2d dv_ap_bp_d = (dv_a_prime + dv_b_prime) / d2;
    math::Vec2d du_a_b_c_d_prime = du_a_b_c * d_prime / d4;
    math::Vec2d dv_a_b_c_d_prime = dv_a_b_c * d_prime / d4;
    double du_c_prime_d = du_c_prime / d2;
    double dv_c_prime_d = dv_c_prime / d2;

    math::Vec2d du_ap_bp_d_du_a_b_c_d_prime = du_ap_bp_d - du_a_b_c_d_prime;
    math::Vec2d dv_ap_bp_d_dv_a_b_c_d_prime = dv_ap_bp_d - dv_a_b_c_d_prime;

    math::Vec2d du_dn;
    math::Vec2d dv_dn;

    /* n00 */
    for (int i = 0; i < 4; ++i)
    {
        du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn00[0 + i];
        dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn00[0 + i];
        jac_dn00[i][0] = (du_dn[0] + du_c_prime_d * dn00[4 + i])
            * grad[0] + (dv_dn[0] + dv_c_prime_d * dn00[4 + i]) * grad[1];
        jac_dn00[i][1] = (du_dn[1] + du_c_prime_d * dn00[8 + i])
            * grad[0] + (dv_dn[1] + dv_c_prime_d * dn00[8 + i]) * grad[1];
    }
    /* n10 */
    for (int i = 0; i < 4; ++i)
    {
        du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn10[0 + i];
        dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn10[0 + i];
        jac_dn10[i][0] = (du_dn[0] + du_c_prime_d * dn10[4 + i])
            * grad[0] + (dv_dn[0] + dv_c_prime_d * dn10[4 + i]) * grad[1];
        jac_dn10[i][1] = (du_dn[1] + du_c_prime_d * dn10[8 + i])
            * grad[0] + (dv_dn[1] + dv_c_prime_d * dn10[8 + i]) * grad[1];
    }
    /* n01 */
    for (int i = 0; i < 4; ++i)
    {
        du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn01[0 + i];
        dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn01[0 + i];
        jac_dn01[i][0] = (du_dn[0] + du_c_prime_d * dn01[4 + i])
            * grad[0] + (dv_dn[0] + dv_c_prime_d * dn01[4 + i]) * grad[1];
        jac_dn01[i][1] = (du_dn[1] + du_c_prime_d * dn01[8 + i])
            * grad[0] + (dv_dn[1] + dv_c_prime_d * dn01[8 + i]) * grad[1];
    }
    /* n11 */
    for (int i = 0; i < 4; ++i)
    {
        du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn11[0 + i];
        dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn11[0 + i];
        jac_dn11[i][0] = (du_dn[0] + du_c_prime_d * dn11[4 + i])
            * grad[0] + (dv_dn[0] + dv_c_prime_d * dn11[4 + i]) * grad[1];
        jac_dn11[i][1] = (du_dn[1] + du_c_prime_d * dn11[8 + i])
            * grad[0] + (dv_dn[1] + dv_c_prime_d * dn11[8 + i]) * grad[1];
    }
}

void
Correspondence::fill_jacobian_derivative_grad(double const* grad,
    double const* dn, math::Vec2d * jac_dn) const
{
    double w2 = w * w;
    double d4 = d2 * d2;
    double d_prime = 2.0 * d * r;

    math::Vec2d du_a_temp = p_prime * r - p * r_prime;
    math::Vec2d du_a = w2 * du_a_temp;
    math::Vec2d du_a_prime = 2.0 * w * du_a_temp;

    math::Vec2d du_b_prime = (p_prime * t[2] - r_prime * t[0]);
    math::Vec2d du_b = w * du_b_prime;

    double du_c_prime = (p * t[2] - r * t[0]);
    math::Vec2d du_c = w_prime * du_c_prime;

    math::Vec2d dv_a_temp = q_prime * r - q * r_prime;
    math::Vec2d dv_a = w2 * dv_a_temp;
    math::Vec2d dv_a_prime = 2.0 * w * dv_a_temp;

    math::Vec2d dv_b_prime = (q_prime * t[2] - r_prime * t[1]);
    math::Vec2d dv_b = w * dv_b_prime;

    double dv_c_prime = (q * t[2] - r * t[1]);
    math::Vec2d dv_c = w_prime * dv_c_prime;

    math::Vec2d du_a_b_c = du_a + du_b + du_c;
    math::Vec2d dv_a_b_c = dv_a + dv_b + dv_c;

    math::Vec2d du_ap_bp_d = (du_a_prime + du_b_prime) / d2;
    math::Vec2d dv_ap_bp_d = (dv_a_prime + dv_b_prime) / d2;
    math::Vec2d du_a_b_c_d_prime = du_a_b_c * d_prime / d4;
    math::Vec2d dv_a_b_c_d_prime = dv_a_b_c * d_prime / d4;
    double du_c_prime_d = du_c_prime / d2;
    double dv_c_prime_d = dv_c_prime / d2;

    math::Vec2d du_ap_bp_d_du_a_b_c_d_prime = du_ap_bp_d - du_a_b_c_d_prime;
    math::Vec2d dv_ap_bp_d_dv_a_b_c_d_prime = dv_ap_bp_d - dv_a_b_c_d_prime;

    math::Vec2d du_dn;
    math::Vec2d dv_dn;

    for (int n = 0; n < 4; ++n)
        for (int i = 0; i < 4; ++i)
        {
            int offset = n * 24;
            du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn[offset + 0 + i];
            dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn[offset + 0 + i];
            jac_dn[n * 4 + i][0] =
                (du_dn[0] + du_c_prime_d * dn[offset + 4 + i]) * grad[0]
                + (dv_dn[0] + dv_c_prime_d * dn[offset + 4 + i]) * grad[1];
            jac_dn[n * 4 + i][1] =
                (du_dn[1] + du_c_prime_d * dn[offset + 8 + i]) * grad[0]
                + (dv_dn[1] + dv_c_prime_d * dn[offset + 8 + i]) * grad[1];
        }
}

void
Correspondence::fill_jacobian_derivative(
    double const* dn, math::Matrix2d * jac_dn) const
{
    double w2 = w * w;
    double d4 = d2 * d2;
    double d_prime = 2.0 * d * r;

    math::Vec2d du_a_temp = p_prime * r - p * r_prime;
    math::Vec2d du_a = w2 * du_a_temp;
    math::Vec2d du_a_prime = 2.0 * w * du_a_temp;

    math::Vec2d du_b_prime = (p_prime * t[2] - r_prime * t[0]);
    math::Vec2d du_b = w * du_b_prime;

    double du_c_prime = (p * t[2] - r * t[0]);
    math::Vec2d du_c = w_prime * du_c_prime;

    math::Vec2d dv_a_temp = q_prime * r - q * r_prime;
    math::Vec2d dv_a = w2 * dv_a_temp;
    math::Vec2d dv_a_prime = 2.0 * w * dv_a_temp;

    math::Vec2d dv_b_prime = (q_prime * t[2] - r_prime * t[1]);
    math::Vec2d dv_b = w * dv_b_prime;

    double dv_c_prime = (q * t[2] - r * t[1]);
    math::Vec2d dv_c = w_prime * dv_c_prime;

    math::Vec2d du_a_b_c = du_a + du_b + du_c;
    math::Vec2d dv_a_b_c = dv_a + dv_b + dv_c;

    math::Vec2d du_ap_bp_d = (du_a_prime + du_b_prime) / d2;
    math::Vec2d dv_ap_bp_d = (dv_a_prime + dv_b_prime) / d2;
    math::Vec2d du_a_b_c_d_prime = du_a_b_c * d_prime / d4;
    math::Vec2d dv_a_b_c_d_prime = dv_a_b_c * d_prime / d4;
    double du_c_prime_d = du_c_prime / d2;
    double dv_c_prime_d = dv_c_prime / d2;

    math::Vec2d du_ap_bp_d_du_a_b_c_d_prime = du_ap_bp_d - du_a_b_c_d_prime;
    math::Vec2d dv_ap_bp_d_dv_a_b_c_d_prime = dv_ap_bp_d - dv_a_b_c_d_prime;

    math::Vec2d du_dn;
    math::Vec2d dv_dn;

    for (int n = 0; n < 4; ++n)
        for (int i = 0; i < 4; ++i)
        {
            int offset = n * 24;
            du_dn = (du_ap_bp_d_du_a_b_c_d_prime) * dn[offset + 0 + i];
            dv_dn = (dv_ap_bp_d_dv_a_b_c_d_prime) * dn[offset + 0 + i];
            jac_dn[n * 4 + i][0] =
                (du_dn[0] + du_c_prime_d * dn[offset + 4 + i]);
            jac_dn[n * 4 + i][1] =
                (dv_dn[0] + dv_c_prime_d * dn[offset + 4 + i]);
            jac_dn[n * 4 + i][2] =
                (du_dn[1] + du_c_prime_d * dn[offset + 8 + i]);
            jac_dn[n * 4 + i][3] =
                (dv_dn[1] + dv_c_prime_d * dn[offset + 8 + i]);
        }
}

SMVS_NAMESPACE_END
