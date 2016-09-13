/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "surface_patch.h"
#include "surface_derivative.h"

SMVS_NAMESPACE_BEGIN

void
SurfacePatch::fill_depth_map (mve::FloatImage & image)
{
    if (this->patch == nullptr)
        this->patch = BicubicPatch::create(this->n00, this->n10,
            this->n01, this->n11);

    std::vector<math::Vec2d> pixels;
    std::vector<double> depths;
    this->fill_values_at_pixels(&pixels, &depths);

    for (std::size_t i = 0; i < pixels.size(); ++i)
        image.at(pixels[i][0], pixels[i][1], 0) = depths[i];
}

void
SurfacePatch::fill_normal_map(mve::FloatImage & image, float inv_flen)
{
    if (this->patch == nullptr)
        this->patch = BicubicPatch::create(this->n00, this->n10,
           this->n01, this->n11);

    std::vector<math::Vec2d> pixels;
    std::vector<double> depths;
    std::vector<math::Vec2d> depth_derivatives;
    this->fill_values_at_pixels(&pixels, &depths, & depth_derivatives);

    for (std::size_t i = 0; i < pixels.size(); ++i)
    {
        math::Vec3d normal;
        double x = pixels[i][0] + 0.5 -
            static_cast<double>(image.width()) / 2.0;
        double y = pixels[i][1] + 0.5 -
            static_cast<double>(image.height()) / 2.0;

        surfderiv::fill_normal(x, y, inv_flen, depths[i],
            depth_derivatives[i][0], depth_derivatives[i][1], *normal);
        for (int c = 0; c < 3; ++c)
            image.at(pixels[i][0], pixels[i][1], c) = normal[c];
    }
}

void
SurfacePatch::fill_values_at_pixels(std::vector<math::Vec2d> * pixels,
    std::vector<double> * depths, std::vector<math::Vec2d> * first_deriv,
    std::vector<math::Vec3d> * second_deriv,
    std::vector<std::size_t> * pids,
    int subsample)
{
    if (this->patch == nullptr)
        this->patch = BicubicPatch::create(this->n00, this->n10,
            this->n01, this->n11);

    pixels->resize(MATH_POW2(this->size) / MATH_POW2(subsample));
    if (depths != nullptr)
        depths->resize(pixels->size());
    if (first_deriv != nullptr)
        first_deriv->resize(pixels->size());
    if (second_deriv != nullptr)
        second_deriv->resize(pixels->size());
    if (pids != nullptr)
        pids->resize(MATH_POW2(this->size) / MATH_POW2(subsample));

    int id = 0;
    for (int pid = 0; pid < this->size * this->size;)// pid += subsample)
    {
        int i = pid % this->size;
        int j = pid / this->size;
        double x = static_cast<double>(i);
        double y = static_cast<double>(j);

        pixels->at(id)[0] = x + static_cast<double>(this->pixel_x);
        pixels->at(id)[1] = y + static_cast<double>(this->pixel_y);
        if (pids != nullptr)
            pids->at(id) = pid;

        x += 0.5;
        y += 0.5;
        x /= this->size;
        y /= this->size;
        if (depths != nullptr)
            depths->at(id) = this->patch->evaluate_f(x, y);
        if (first_deriv != nullptr)
        {
            first_deriv->at(id)[0] = this->patch->evaluate_dx(x, y);
            first_deriv->at(id)[1] = this->patch->evaluate_dy(x, y);
            first_deriv->at(id) /= this->size;
        }
        if (second_deriv != nullptr)
        {
            second_deriv->at(id)[0] = this->patch->evaluate_dxy(x, y);
            second_deriv->at(id)[1] = this->patch->evaluate_dxx(x, y);
            second_deriv->at(id)[2] = this->patch->evaluate_dyy(x, y);
            second_deriv->at(id) /= (this->size * this->size);
        }
        id += 1;
        if (subsample > 1)
        {
            pid += subsample;
            if ((pid/this->size) % (subsample) == 1)
                pid += this->size * (subsample - 1);
        } else {
            pid += 1;
        }
    }
}

void
SurfacePatch::fill_values_at_nodes (std::vector<math::Vec2d> * pixels,
    std::vector<double> * depths, std::vector<math::Vec2d> * first_deriv,
    std::vector<double> * second_deriv)
{
    if  (pixels != nullptr)
    {
        pixels->resize(4);
        pixels->at(0)[0] = this->pixel_x;
        pixels->at(0)[1] = this->pixel_y;
        pixels->at(1)[0] = this->pixel_x + this->size;
        pixels->at(1)[1] = this->pixel_y;
        pixels->at(2)[0] = this->pixel_x;
        pixels->at(2)[1] = this->pixel_y + this->size;
        pixels->at(3)[0] = this->pixel_x + this->size;
        pixels->at(3)[1] = this->pixel_y + this->size;
    }

    if (depths != nullptr)
    {
        depths->resize(4);
        depths->at(0) = this->n00->f;
        depths->at(1) = this->n10->f;
        depths->at(2) = this->n01->f;
        depths->at(3) = this->n11->f;
    }
    if (first_deriv != nullptr)
    {
        first_deriv->resize(4);
        first_deriv->at(0)[0] = this->n00->dx / this->size;
        first_deriv->at(1)[0] = this->n10->dx / this->size;
        first_deriv->at(2)[0] = this->n01->dx / this->size;
        first_deriv->at(3)[0] = this->n11->dx / this->size;

        first_deriv->at(0)[1] = this->n00->dy / this->size;
        first_deriv->at(1)[1] = this->n10->dy / this->size;
        first_deriv->at(2)[1] = this->n01->dy / this->size;
        first_deriv->at(3)[1] = this->n11->dy / this->size;
    }
    if (second_deriv != nullptr)
    {
        second_deriv->resize(4);
        second_deriv->at(0) = this->n00->dxy / this->size;
        second_deriv->at(1) = this->n10->dxy / this->size;
        second_deriv->at(2) = this->n01->dxy / this->size;
        second_deriv->at(3) = this->n11->dxy / this->size;
    }
}

SMVS_NAMESPACE_END
