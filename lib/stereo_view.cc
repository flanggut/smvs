/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "mve/image_tools.h"

#include "stereo_view.h"

SMVS_NAMESPACE_BEGIN

StereoView::StereoView (mve::View::Ptr view,
    std::string const& image_embedding)
    : view(view), image_embedding(image_embedding)
{
    this->image = mve::image::byte_to_float_image(
        this->view->get_byte_image(this->image_embedding));
    /* use only for debug */
#if SMVS_DEBUG
    this->debug = mve::FloatImage::create(this->image->width(),
        this->image->height(), 3);
#endif
}

void
StereoView::set_scale(int scale, bool debug)
{
    mve::FloatImage::Ptr fimage = this->image->duplicate();

    double sigma = 0.12 * std::pow(2.0, scale) + 0.2;
    fimage = mve::image::blur_gaussian<float>(fimage, sigma);

    this->initialize_image_gradients(fimage);

    if (debug)
    {
        mve::ByteImage::Ptr blur = mve::image::float_to_byte_image(fimage);
        this->view->set_image(blur, "smvs-image");
        this->view->set_image(this->image_grad, "smvs-gradients");
        this->view->set_image(this->image_hessian, "smvs-hessian");
        this->view->save_view();
    }

    this->view->cache_cleanup();
}

void
StereoView::initialize_image_gradients (mve::FloatImage::ConstPtr image)
{
    if(image->channels() > 1)
        image = mve::image::desaturate<float>(image,
             mve::image::DESATURATE_LUMINANCE);

    this->image_grad = mve::FloatImage::create(image->width(),
        image->height(), 2);
    this->image_hessian = mve::FloatImage::create(image->width(),
        image->height(), 3);

    this->compute_gradients_and_hessian(image, this->image_grad,
        this->image_hessian);
}

void
StereoView::initialize_linear (bool gamma_correction)
{
    if(this->image->channels() > 1)
        this->linear_image = mve::image::desaturate<float>(this->image,
            mve::image::DESATURATE_LUMINANCE);
    else
        this->linear_image = this->image->duplicate();

    if (gamma_correction)
        mve::image::gamma_correct_inv_srgb<float>(this->linear_image);

    this->linear_grad = mve::FloatImage::create(this->linear_image->width(),
        this->linear_image->height(), 2);
    this->compute_gradients_and_hessian(this->linear_image, this->linear_grad);
}

mve::FloatImage::Ptr
StereoView::get_shading_image (void)
{
    mve::FloatImage::Ptr shading = this->linear_image->duplicate();
    return shading;
}

mve::ByteImage::ConstPtr
StereoView::get_byte_image (void) const
{
    mve::ByteImage::Ptr byte_image =
        this->view->get_byte_image(this->image_embedding);
    if(byte_image->channels() > 1)
        byte_image = mve::image::desaturate<uint8_t>(byte_image,
            mve::image::DESATURATE_LUMINANCE);
    return byte_image;
}

void
StereoView::compute_gradients_and_hessian (mve::FloatImage::ConstPtr input,
    mve::FloatImage::Ptr gradient, mve::FloatImage::Ptr hessian)
{
    gradient->fill(0.0);
    if (hessian != nullptr)
        hessian->fill(0.0);

    math::Matrix<double, 6, 9> M;
    M(0,0) = 1.0 / 6.0;
    M(0,1) = 1.0 / 6.0;
    M(0,2) = 1.0 / 6.0;
    M(0,3) = -1.0 / 3.0;
    M(0,4) = -1.0 / 3.0;
    M(0,5) = -1.0 / 3.0;
    M(0,6) = 1.0 / 6.0;
    M(0,7) = 1.0 / 6.0;
    M(0,8) = 1.0 / 6.0;

    M(1,0) = 1.0 / 6.0;
    M(1,1) = -1.0 / 3.0;
    M(1,2) = 1.0 / 6.0;
    M(1,3) = 1.0 / 6.0;
    M(1,4) = -1.0 / 3.0;
    M(1,5) = 1.0 / 6.0;
    M(1,6) = 1.0 / 6.0;
    M(1,7) = -1.0 / 3.0;
    M(1,8) = 1.0 / 6.0;

    M(2,0) = 1.0 / 4.0;
    M(2,1) = 0.0;
    M(2,2) = -1.0/ 4.0;
    M(2,3) = 0.0;
    M(2,4) = 0.0;
    M(2,5) = 0.0;
    M(2,6) = -1.0/ 4.0;
    M(2,7) = 0.0;
    M(2,8) = 1.0/ 4.0;

    M(3,0) = -1.0 / 6.0;
    M(3,1) = -1.0 / 6.0;
    M(3,2) = -1.0 / 6.0;
    M(3,3) = 0.0;
    M(3,4) = 0.0;
    M(3,5) = 0.0;
    M(3,6) = 1.0 / 6.0;
    M(3,7) = 1.0 / 6.0;
    M(3,8) = 1.0 / 6.0;

    M(4,0) = -1.0 / 6.0;
    M(4,1) = 0.0;
    M(4,2) = 1.0 / 6.0;
    M(4,3) = -1.0 / 6.0;
    M(4,4) = 0.0;
    M(4,5) = 1.0 / 6.0;
    M(4,6) = -1.0 / 6.0;
    M(4,7) = 0.0;
    M(4,8) = 1.0 / 6.0;

    M(5,0) = -1.0 / 9.0;
    M(5,1) = 2.0 / 9.0;
    M(5,2) = -1.0 / 9.0;
    M(5,3) = 2.0 / 9.0;
    M(5,4) = 5.0 / 9.0;
    M(5,5) = 2.0 / 9.0;
    M(5,6) = -1.0 / 9.0;
    M(5,7) = 2.0 / 9.0;
    M(5,8) = -1.0 / 9.0;


    for (int y = 1; y < input->height() - 1; ++y)
        for (int x = 1; x < input->width() - 1; ++x)
        {
            math::Vector<double, 9> v;
            int c = 0;
            for (int a = -1; a < 2; ++a)
                for (int b = -1; b < 2; ++b)
                    v[c++] = input->at(x + a, y + b, 0);

            math::Vector<double, 6> r;
            r = M * v;

            gradient->at(x, y, 0) = r[3];
            gradient->at(x, y, 1) = r[4];

            if (hessian == nullptr)
                continue;
            hessian->at(x, y, 0) = 2.0 * r[0];
            hessian->at(x, y, 1) = r[2];
            hessian->at(x, y, 2) = 2.0 * r[1];
        }
}

SMVS_NAMESPACE_END
