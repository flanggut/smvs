/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "math/matrix_svd.h"

#include "light_optimizer.h"
#include "spherical_harmonics.h"

SMVS_NAMESPACE_BEGIN

LightOptimizer::LightOptimizer (Surface::Ptr surface, StereoView::Ptr view)
    : surface(surface), view(view)
{
}

GlobalLighting::Ptr
LightOptimizer::fit_lighting_to_image (mve::FloatImage::ConstPtr image)
{
    math::Vector<double, 16> b;
    b.fill(0.0);
    math::Matrix<double, 16, 16> A;
    A.fill(0.0);

    mve::FloatImage::Ptr normals = this->surface->get_normal_map(
        this->view->get_inverse_flen());
    for (int p = 0; p < normals->get_pixel_amount(); ++p)
    {
        math::Vec3d normal(normals->at(p, 0), normals->at(p, 1),
            normals->at(p, 2));
        if (std::fabs(normal.norm() - 1.0) > 1e-6
            || image->at(p) < 0.05f)
            continue;

        GlobalLighting::SHBasis sh;
        sh::evaluate_4_band(*normal, *sh);

        for (int i = 0; i < 16; ++i)
        {
            b[i] += sh[i] * image->at(p);
            for (int j = 0; j < 16; ++j)
                A(j,i) += sh[i] * sh[j];
        }
    }
    math::Matrix<double, 16, 16> A_inv;
    math::matrix_pseudo_inverse(A, &A_inv);
    GlobalLighting::Params lighting_params = A_inv * b;
    
    return GlobalLighting::create(lighting_params);
}

SMVS_NAMESPACE_END
