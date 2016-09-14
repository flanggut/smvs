/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_BICUBIC_PATCH_HEADER
#define SMVS_BICUBIC_PATCH_HEADER

#include <memory>
#include <vector>

#include "defines.h"

SMVS_NAMESPACE_BEGIN

class BicubicPatch
{
public:
    struct Node
    {
    public:
        typedef std::shared_ptr<Node> Ptr;
        typedef std::shared_ptr<Node const> ConstPtr;

        /* Function value */
        double f = 0.0;
        /* Derivative in x-direction */
        double dx = 0.0;
        /* Derivative in y-direction */
        double dy = 0.0;
        /* Mixed second derivative */
        double dxy = 0.0;

        static Node::Ptr create (void)
        {
            return std::shared_ptr<Node>(new Node());
        }
        
    private:
        Node (void) = default;
    };

public:
    typedef std::shared_ptr<BicubicPatch> Ptr;
    typedef std::shared_ptr<BicubicPatch const> ConstPtr;

    static Ptr create(Node::ConstPtr node00, Node::ConstPtr node10,
        Node::ConstPtr node01, Node::ConstPtr node11);

    static Ptr fit_to_data(double const* x, double const* y,
        double const* data, int size);

    double evaluate_f (double x, double y) const;
    double evaluate_dx (double x, double y) const;
    double evaluate_dy (double x, double y) const;
    double evaluate_dxy (double x, double y) const;
    void evaluate_all (double x, double y, double * values) const;

    /* These values are not continuous across patch borders */
    double evaluate_dxx (double x, double y) const;
    double evaluate_dyy (double x, double y) const;
    double evaluate_dxx (double const* x, double const* y) const;
    double evaluate_dyy (double const* x, double const* y) const;

    static void node_derivatives (double x, double y, double * d_00,
        double * d_10, double * d_01, double * d_11);

    static void node_derivatives_for_patchsize (double x, double y,
        double patchsize, double * d_00, double * d_10,
        double * d_01, double * d_11);

private:
    BicubicPatch (void);
    BicubicPatch (double const* coefficients);
    BicubicPatch (Node::ConstPtr node00, Node::ConstPtr node10,
        Node::ConstPtr node01, Node::ConstPtr node11);

    void compute_coefficients (void);
    void compute_coefficients_sanity_check (void);
    void compute_coefficient_derivatives (void);

    double evaluate_f (double const* x, double const* y) const;
    double evaluate_dx (double const* x, double const* y) const;
    double evaluate_dy (double const* x, double const* y) const;
    double evaluate_dxy (double const* x, double const* y) const;
    void evaluate_all (double const* x, double const* y, double* values) const;

private:
    Node::ConstPtr n00;
    Node::ConstPtr n10;
    Node::ConstPtr n01;
    Node::ConstPtr n11;

    double coeffs[4][4];
};

/* ------------------------ Implementation ------------------------ */

inline BicubicPatch::Ptr
BicubicPatch::create(Node::ConstPtr node00, Node::ConstPtr node10,
    Node::ConstPtr node01, Node::ConstPtr node11)
{
    return Ptr(new BicubicPatch(node00, node10, node01, node11));
}

inline
BicubicPatch::BicubicPatch (Node::ConstPtr node00, Node::ConstPtr node10,
    Node::ConstPtr node01, Node::ConstPtr node11)
    : n00(node00), n10(node10), n01(node01), n11(node11)
{
    this->compute_coefficients();
}

inline
BicubicPatch::BicubicPatch (double const* coefficients)
{
    std::copy(coefficients, coefficients + 16, this->coeffs[0]);
}

SMVS_NAMESPACE_END

#endif /* SMVS_BICUBIC_PATCH_HEADER */
