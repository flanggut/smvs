/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_CONJUGATE_GRADIENT_HEADER
#define SMVS_CONJUGATE_GRADIENT_HEADER

#include "sse_vector.h"

SMVS_NAMESPACE_BEGIN

class ConjugateGradient
{
public:
    typedef SSEVector Vector;

    enum ReturnInfo
    {
        CG_CONVERGENCE,
        CG_MAX_ITERATIONS,
        CG_INVALID_INPUT
    };

    struct Options
    {
        Options (void);
        int max_iterations;
        double error_tolerance;
        double q_tolerance;
    };

    struct Status
    {
        Status (void);
        int num_iterations;
        ReturnInfo info;
    };

    class Functor
    {
    public:
        virtual Vector multiply (Vector const& x) const = 0;
        virtual std::size_t input_size (void) const = 0;
        virtual std::size_t output_size (void) const = 0;
    };

public:
    ConjugateGradient (Options const& opts);

    Status solve (Functor const& A, Vector const& b, Vector* x,
        Functor const* P = nullptr);

private:
    Options opts;
    Status status;
};

/* ------------------------ Implementation ------------------------ */

inline
ConjugateGradient::Options::Options (void)
: max_iterations(1000)
, error_tolerance(1e-20)
, q_tolerance(1e-3)
{
}

inline
ConjugateGradient::Status::Status (void)
: num_iterations(0)
{
}

inline
ConjugateGradient::ConjugateGradient
(Options const& options)
: opts(options)
{
}

inline
ConjugateGradient::Status
ConjugateGradient::solve(Functor const& A, Vector const& b, Vector* x,
    Functor const* P)
{
    if (x == nullptr)
        throw std::invalid_argument("RHS must not be null");

    if (A.output_size() != b.size())
    {
        this->status.info = CG_INVALID_INPUT;
        return this->status;
    }

    /* Set intial x = 0. */
    if (x->size() != A.input_size())
    {
        x->clear();
        x->resize(A.input_size(), 0.0);
    }
    else
    {
        x->fill(0.0);
    }

    /* Initial residual is r = b - Ax with x = 0. */
    Vector r = b;

    /* Regular search direction. */
    Vector d;
    /* Preconditioned search direction. */
    Vector z;
    /* Norm of residual. */
    double r_dot_r;

    /* Compute initial search direction. */
    if (P == nullptr)
    {
        r_dot_r = r.dot(r);
        d = r;
    }
    else
    {
        z = (*P).multiply(r);
        r_dot_r = z.dot(r);
        d = z;
    }

    /* Compute initial value of the quadratic model Q = x'Ax - 2 * b'x. */
    double Q0 = -1.0 * x->dot(b.add(r));

    for (this->status.num_iterations = 1;
         this->status.num_iterations < this->opts.max_iterations;
         this->status.num_iterations += 1)
    {
        /* Compute step size in search direction. */
        Vector Ad = A.multiply(d);
        double alpha = r_dot_r / d.dot(Ad);

        /* Update parameter vector. */
        *x = (*x).add(d.multiply(alpha));

        /* Compute new residual and its norm. */
        r = r.subtract(Ad.multiply(alpha));
        double new_r_dot_r = r.dot(r);

        /* Residual based termination. */
        if (new_r_dot_r < this->opts.error_tolerance)
        {
            this->status.info = CG_CONVERGENCE;
            return this->status;
        }

        /* Quadratic model based termination.
         * From Ceres Solver:
         * For PSD matrices A, let
         *
         *   Q(x) = x'Ax - 2b'x
         *
         * be the cost of the quadratic function defined by A and b. Then,
         * the solver terminates at iteration i if
         *
         *   i * (Q(x_i) - Q(x_i-1)) / Q(x_i) < q_tolerance.
         *
         * This termination criterion is more useful when using CG to
         * solve the Newton step. This particular convergence test comes
         * from Stephen Nash's work on truncated Newton
         * methods. References:
         *
         *   1. Stephen G. Nash & Ariela Sofer, Assessing A Search
         *   Direction Within A Truncated Newton Method, Operation
         *   Research Letters 9(1990) 219-221.
         *
         *   2. Stephen G. Nash, A Survey of Truncated Newton Methods,
         *   Journal of Computational and Applied Mathematics,
         *   124(1-2), 45-59, 2000.
         *
         */
        const double Q1 = -1.0 * x->dot(b.add(r));
        const double zeta = this->status.num_iterations * (Q1 - Q0) / Q1;
        if (zeta < this->opts.q_tolerance)
        {
            this->status.info = CG_CONVERGENCE;
            return this->status;
        }
        Q0 = Q1;

        /* Precondition residual if necessary. */
        if (P != nullptr)
        {
            z = P->multiply(r);
            new_r_dot_r = z.dot(r);
        }

        /*
         * Update search direction.
         * The next residual will be orthogonal to new Krylov space.
         */
        double beta = new_r_dot_r / r_dot_r;
        if (P != nullptr)
            d = z.add(d.multiply(beta));
        else
            d = r.add(d.multiply(beta));

        /* Update residual norm. */
        r_dot_r = new_r_dot_r;
    }

    this->status.info = CG_MAX_ITERATIONS;
    return this->status;
}

SMVS_NAMESPACE_END

#endif /* SMVS_CONJUGATE_GRADIENT_HEADER */
