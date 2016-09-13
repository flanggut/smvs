/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "global_lighting.h"
#include "spherical_harmonics.h"

SMVS_NAMESPACE_BEGIN

double
GlobalLighting::value_for_normal (math::Vec3d const& normal) const
{
    SHBasis sh_basis;
    sh::evaluate_4_band(*normal, *sh_basis);
    return this->parameters.dot(sh_basis);
}

mve::FloatImage::Ptr
get_sphere_normals (int dim)
{
    if (dim % 2 == 0)
        dim += 1;

    mve::FloatImage::Ptr normals =
        mve::FloatImage::create(dim,dim,3);
    normals->fill(0);
    for (int x = 0; x < normals->width(); ++x)
        for (int y = 0; y < normals->height(); ++y)
        {
            math::Vec3f normal;
            normal[0] = 2.0f * (float)(x - normals->width() / 2 ) /
                (float) normals->width();
            normal[1] = 2.0f * (float)(normals->height() / 2 - y) /
                (float) normals->height();
            if ((MATH_POW2(normal[0]) + MATH_POW2(normal[1])) > 1.)
                continue;
            normal[2] = std::sqrt(1 - MATH_POW2(normal[0]) -
                MATH_POW2(normal[1]));
            normals->at(x,y,0) = normal[0];
            normals->at(x,y,1) = normal[1];
            normals->at(x,y,2) = normal[2];
        }
    return normals;
}

mve::FloatImage::Ptr
GlobalLighting::get_rendered_sphere (int dim) const
{
    mve::FloatImage::Ptr normals = get_sphere_normals(dim);
    return this->render_normal_map(normals);
}

mve::FloatImage::Ptr
GlobalLighting::render_normal_map (mve::FloatImage::ConstPtr normals) const
{
    mve::FloatImage::Ptr image = mve::FloatImage::create(normals->width(),
        normals->height(), 1);
    image->fill(0.0f);

    for (int i = 0; i < normals->get_pixel_amount(); ++i)
    {
        math::Vec3d normal(normals->at(i, 0), normals->at(i, 1),
            normals->at(i, 2));
        if (normal.norm() == 0.0)
            continue;
        image->at(i, 0) = this->value_for_normal(normal);
    }

    return image;
}

SMVS_NAMESPACE_END
