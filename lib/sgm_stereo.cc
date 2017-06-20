/*
 * Copyright (c) 2016-2017, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <smmintrin.h> // SSE4_1
#if defined(_WIN32)
#   include <nmmintrin.h> // for hamming distance
#else // Linux, OSX, ...
#   include <popcntintrin.h> // for hamming distance
#endif

#include "mve/depthmap.h"
#include "mve/image_tools.h"
#include "util/timer.h"

#include "sgm_stereo.h"
#include "correspondence.h"

SMVS_NAMESPACE_BEGIN

SGMStereo::SGMStereo (Options const& opts, StereoView::Ptr main,
    StereoView::Ptr neighbor)
    : opts(opts), main(main), neighbor(neighbor)
{
    this->main_image = this->main->get_byte_image();
    for (int i = 0; i < this->opts.scale; ++i)
        this->main_image = mve::image::rescale_half_size<uint8_t>(
            this->main_image);

    this->neighbor_image = this->neighbor->get_byte_image();
    for (int i = 0; i < this->opts.scale; ++i)
        this->neighbor_image = mve::image::rescale_half_size<uint8_t>(
            this->neighbor_image);

    this->cost_updates.resize(opts.num_steps);
    this->min_cost_updates.resize(opts.num_steps);
    this->mins.resize(opts.num_steps);
}

mve::FloatImage::Ptr
SGMStereo::reconstruct (SGMStereo::Options sgm_opts, StereoView::Ptr main_view,
    StereoView::Ptr neighbor, mve::Bundle::ConstPtr bundle)
{
    float depth_range[2];
    depth_range[0] = sgm_opts.min_depth;
    depth_range[1] = sgm_opts.max_depth;

    if (bundle != nullptr && sgm_opts.max_depth == 0.0)
        fill_depth_range_for_view(bundle, main_view, depth_range);
    SGMStereo sgm1(sgm_opts, main_view, neighbor);
    mve::FloatImage::Ptr d_main = sgm1.run_sgm(depth_range[0], depth_range[1]);

    if (bundle != nullptr && sgm_opts.max_depth == 0.0)
        fill_depth_range_for_view(bundle, neighbor, depth_range);
    SGMStereo sgm2(sgm_opts, neighbor, main_view);
    mve::FloatImage::Ptr d_neig = sgm2.run_sgm(depth_range[0], depth_range[1]);

    mve::CameraInfo const &main_cam = main_view->get_camera();
    mve::CameraInfo const &neighbor_cam = neighbor->get_camera();
    math::Matrix3f M;
    math::Vec3f t;
    main_cam.fill_reprojection(neighbor_cam, d_main->width(),
        d_main->height(), d_neig->width(), d_neig->height(), *M, *t);

    int const cut = 0.03 * std::max(d_neig->width(), d_neig->height());
    for (int x = 0; x < d_main->width(); ++x)
        for (int y = 0; y < d_main->height(); ++y)
        {
            if (d_main->at(x, y, 0) == 0)
                continue;
            Correspondence c(M, t, x, y, d_main->at(x, y, 0));
            math::Vec2d coords;
            c.fill(*coords);
            if (coords[0] < cut || coords[0] >= d_neig->width() - cut
                || coords[1] < cut || coords[1] >= d_neig->height() - cut)
            {
                d_main->at(x, y, 0) = 0;
                continue;
            }
            float cdepth = c.get_depth();
            float ndepth = d_neig->at(coords[0], coords[1], 0);
            float ratio = std::min(cdepth, ndepth) / std::max(cdepth, ndepth);
            if (ndepth == 0 || ratio < 0.93)
                d_main->at(x, y, 0) = 0;
        }
    if (sgm_opts.debug_lvl > 1)
        std::cout << "SGM finished." << std::endl;
    
    return d_main;
}

mve::FloatImage::Ptr
SGMStereo::run_sgm (float min_depth, float max_depth)
{
    if (this->opts.debug_lvl > 1)
        std::cout << "Running SGM width depth range: [" << min_depth
            << " ; " << max_depth << "]" << std::endl;

    util::WallTimer timer;

    /* Create Census cost volume */
    this->create_cost_volume(min_depth, max_depth, this->opts.num_steps);
    if (this->opts.debug_lvl > 1)
        std::cout << "Building Cost Volume took: "
            << timer.get_elapsed_sec() << "s" << std::endl;


    /* Run SGM cost aggregation */
    timer.reset();
    this->aggregate_sgm_costs();
    if (this->opts.debug_lvl > 1)
        std::cout << "SGM Cost aggregation took : "
            << timer.get_elapsed_sec() << "s" << std::endl;

    /* Extract final depth */
    mve::FloatImage::Ptr depth = this->depth_from_sgm_volume();
    return depth;
}

void
SGMStereo::census_filter (mve::ByteImage::ConstPtr image,
    mve::Image<uint64_t>::Ptr filtered)
{
    filtered->fill(0);
    for (int x = 4; x < image->width() - 5; ++x)
        for (int y = 3; y < image->height() - 4; ++y)
        {
            if (image->at(x, y, 0) == 0)
                continue;

            uint64_t census = 0;
            for (int i = x - 4; i < x + 5; ++i)
                for (int j = y - 3; j < y + 4; ++j)
                {
                    census *= 2;
                    if (image->at(x, y, 0) < image->at(i, j, 0))
                        census += 1;
                }
            filtered->at(x, y, 0) = census;
        }
}

void
SGMStereo::warped_neighbor_for_depth (float depth, mve::ByteImage::Ptr image)
{
    math::Matrix3f M;
    math::Vec3f t;
    mve::CameraInfo n_cam = this->neighbor->get_camera();
    this->main->get_camera().fill_reprojection(n_cam,
        this->main_image->width(), this->main_image->height(),
        this->neighbor_image->width(), this->neighbor_image->height(), *M, *t);

    for (int x = 0; x < image->width(); ++x)
        for (int y = 0; y < image->height(); ++y)
        {
            math::Vec3f target_pixel(0.5f + x, 0.5f + y, 1.f);
            math::Vec3f projected = M * target_pixel * depth + t;
            projected[0] /= projected[2];
            projected[1] /= projected[2];
            projected[0] -= 0.5f;
            projected[1] -= 0.5f;

            if (projected[0] < 0 || projected[1] < 0
                || projected[0] > this->neighbor_image->width() - 1
                || projected[1] > this->neighbor_image->height() - 1)

                image->at(x,y,0) = 0;
            else
                image->at(x,y,0) = this->neighbor_image->linear_at(
                        projected[0], projected[1], 0);
        }
}

void
SGMStereo::create_cost_volume (float min_depth, float max_depth, int num_steps)
{
    this->cost_volume_depths.clear();
    this->cost_volume_depths.resize(num_steps);
    float inv_depth = 1.0f / max_depth;
    float increment = (1.0f / min_depth - inv_depth) / (num_steps - 1);
    for (int i = 0; i < num_steps; ++i)
    {
        this->cost_volume_depths[i] = 1.0f / inv_depth;
        inv_depth += increment;
    }

    mve::ByteImage::Ptr n_warped = mve::ByteImage::create(
        this->main_image->width(), this->main_image->height(), 1);
    mve::Image<uint64_t>::Ptr n_warped_census =
        mve::Image<uint64_t>::create(n_warped->width(), n_warped->height(), 1);
    mve::Image<uint64_t>::Ptr main_census = mve::Image<uint64_t>::create(
        main_image->width(), main_image->height(), 1);
    this->census_filter(main_image, main_census);

#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
    std::size_t const volume_size = main_census->width() *
        main_census->height() * num_steps;
    this->sse_cost_volume.resize(volume_size, 256);
#else
    this->cost_volume = mve::ByteImage::create(
        main_census->width(), main_census->height(), num_steps);
        this->cost_volume->fill(255);
#endif

    for (int i = 0; i < num_steps; ++i)
    {
        this->warped_neighbor_for_depth(this->cost_volume_depths[i], n_warped);
        this->census_filter(n_warped, n_warped_census);
        for (int p = 0; p < main_census->get_pixel_amount(); ++p)
        {
            if (n_warped->at(p) == 0)
                continue;

            /* hamming distance between census values */
            uint8_t count = _mm_popcnt_u64(
                main_census->at(p) ^ n_warped_census->at(p));

#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
            this->sse_cost_volume.at(p * num_steps + i) = count;
#else
            this->cost_volume->at(p, i) = count;
#endif
        }
    }
}

mve::FloatImage::Ptr
SGMStereo::depth_from_cost_volume (void)
{
    mve::FloatImage::Ptr depthmap = mve::FloatImage::create(
        this->main->get_width(), this->main->get_height(), 1);

    for (int y = 0, p = 0; y < depthmap->height(); ++y)
        for (int x = 0; x < depthmap->width(); ++x, ++p)
        {
            uint8_t min_error = 200;
            std::size_t min_index = 0;
            for (int i = 0; i < (int)this->cost_volume_depths.size(); ++i)
            {
                uint8_t const value = this->cost_volume->at(p, i);
                if (value < min_error)
                {
                    min_error = value;
                    min_index = i;
                }
            }
            if(min_index == 0)
                depthmap->at(p) = 0;
            else
                depthmap->at(p) = this->cost_volume_depths[min_index];
        }
    return depthmap;
}

mve::FloatImage::Ptr
SGMStereo::depth_from_sgm_volume (void)
{
    mve::FloatImage::Ptr depthmap = mve::FloatImage::create(
        this->main_image->width(), this->main_image->height(), 1);

    int const num_steps = this->opts.num_steps;

    for (int y = 0, p = 0; y < depthmap->height(); ++y)
        for (int x = 0; x < depthmap->width(); ++x, ++p)
        {
            uint16_t min_error = std::numeric_limits<uint16_t>::max();
            int min_index = 0;
            for (int i = 0; i < num_steps; ++i)
            {
#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
                uint16_t const value = this->sse_sgm_volume[p * num_steps + i];
#else
                uint16_t const value = this->sgm_volume->at(p, i);
#endif
                if (value < min_error)
                {
                    min_error = value;
                    min_index = i;
                }
            }
            if(min_index < 2 || this->main_image->at(p) < 25)
                depthmap->at(p) = 0;
            else
                depthmap->at(p) = this->cost_volume_depths[min_index];
        }
    return depthmap;
}


void
SGMStereo::fill_path_cost(int x, int y, int px, int py, mve::RawImage::Ptr path)
{
    int const num_steps = path->channels();

    int i1 = this->main_image->at(x, y, 0);
    int i2 = this->main_image->at(px, py, 0);
    uint16_t diff = std::abs(i1 - i2) + 1;
    uint16_t const penalty1 = this->opts.penalty1;
    uint16_t const penalty2 = std::max(penalty1 * 3 / 2,
        this->opts.penalty2 / diff);

    uint16_t sgm_cost;
    uint16_t cost_update;
    uint16_t min_prev_cost = std::numeric_limits<uint16_t>::max();

    for (int i = 0; i < num_steps; ++i)
        min_prev_cost = std::min(path->at(px, py, i), min_prev_cost);

    for (int i = 0; i < num_steps; ++i)
    {
        cost_update = path->at(px, py, i);
        for (int j = 0; j < num_steps; ++j)
        {
            if (i == j)
                continue;
            if (std::abs(j - i) == 1)
                cost_update = std::min<uint16_t>(cost_update,
                    path->at(px, py, j) + penalty1);
            else
                cost_update = std::min<uint16_t>(cost_update,
                    path->at(px, py, j) + penalty2);
        }
        sgm_cost = this->cost_volume->at(x, y, i) + cost_update - min_prev_cost;
        path->at(x, y, i) = sgm_cost;
        this->sgm_volume->at(x, y, i) += sgm_cost;
    }
}

#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
uint16_t
SGMStereo::sse_reduction_min (uint16_t * data, std::size_t size)
{
    __m128i const* data_ptr = reinterpret_cast<__m128i const*>(data);
    __m128i * min_ptr = reinterpret_cast<__m128i*>(this->mins.data());
    *min_ptr = _mm_load_si128(data_ptr++);
    for (std::size_t idx = 8; idx < size; idx += 8, ++data_ptr)
        *min_ptr = _mm_min_epu16(*min_ptr, *data_ptr);
    *min_ptr = _mm_minpos_epu16(*min_ptr);
    return this->mins[0];
}

void
SGMStereo::fill_path_cost_sse (int base, int pbase,
        util::AlignedMemory<uint16_t> * path)
{
    int const d_stride = this->opts.num_steps;
    int i1 = this->main_image->at(base / d_stride);
    int i2 = this->main_image->at(pbase / d_stride);
    uint16_t diff = std::abs(i1 - i2) + 1;
    uint16_t const penalty1 = this->opts.penalty1;
    uint16_t const penalty2 = std::max(penalty1 * 3 / 2,
        this->opts.penalty2 / diff);

    /* Find minmal cost in prev */
    uint16_t min_prev_cost = sse_reduction_min(&path->at(pbase), d_stride);

    for (int idx = 0; idx < d_stride; idx += 1)
    {
        __m128i pen = _mm_set1_epi16(penalty2);
        __m128i * path_ptr = reinterpret_cast<__m128i *>(&path->at(pbase));
        __m128i * cu_ptr = reinterpret_cast<__m128i *>(cost_updates.data());
        for (int idx2 = 0; idx2 < d_stride; idx2 += 8, ++path_ptr, ++cu_ptr)
            *cu_ptr = _mm_add_epi16(*path_ptr, pen);

        this->cost_updates[idx] = path->at(pbase + idx);
        if (idx > 0)
            this->cost_updates[idx - 1] = path->at(pbase + idx - 1) + penalty1;
        if (idx < d_stride - 1)
            this->cost_updates[idx + 1] = path->at(pbase + idx + 1) + penalty1;

        min_cost_updates[idx] = sse_reduction_min(&this->cost_updates[0],
            d_stride);
    }

    __m128i min_prev = _mm_set1_epi16(min_prev_cost);
    __m128i const* cost_vol_ptr = reinterpret_cast<__m128i const*>(
        &this->sse_cost_volume[base]);
    __m128i const* cost_upd_ptr = reinterpret_cast<__m128i const*>(
        &min_cost_updates[0]);
    __m128i * sgm_vol_ptr = reinterpret_cast<__m128i *>(
        &this->sse_sgm_volume[base]);
    __m128i * path_ptr = reinterpret_cast<__m128i *>(&path->at(base));

    for (int idx = 0; idx < d_stride; idx += 8, ++cost_vol_ptr, ++cost_upd_ptr,
         ++sgm_vol_ptr, ++path_ptr)
    {
        *path_ptr = _mm_add_epi16(*cost_vol_ptr, *cost_upd_ptr);
        *path_ptr = _mm_sub_epi16(*path_ptr, min_prev);
        *sgm_vol_ptr = _mm_add_epi16(*sgm_vol_ptr, *path_ptr);
    }
}

void
SGMStereo::copy_cost_and_add_to_sgm(util::AlignedMemory<uint16_t> *local_volume,
    int base)
{
    int const d_stride = this->opts.num_steps;
    std::copy(&this->sse_cost_volume[base], &this->sse_cost_volume[base +
        d_stride], &local_volume->at(base));

    for (int disp = 0; disp < d_stride; disp += 8)
    {
        __m128i cost = _mm_loadu_si128(reinterpret_cast<__m128i const*>(
            &this->sse_cost_volume[base + disp]));
        __m128i sgm = _mm_loadu_si128(reinterpret_cast<__m128i const*>(
            &this->sse_sgm_volume[base + disp]));
        __m128i sum = _mm_add_epi16(sgm, cost);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(
            &this->sse_sgm_volume[base + disp]), sum);
    }
}
#endif

void
SGMStereo::aggregate_sgm_costs (void)
{
    int const width = this->main_image->width();
    int const height = this->main_image->height();

#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
    int const d_stride = this->opts.num_steps;
    int const y_stride = this->main_image->width();
    this->sse_sgm_volume.resize(this->sse_cost_volume.size(), 0);
    this->mins.resize(8);
    util::AlignedMemory<uint16_t> sse_local_volume(
        this->sse_cost_volume.size(), 0);
    util::AlignedMemory<uint16_t> sse_local_volume_d1(
        this->sse_cost_volume.size(), 0);
    util::AlignedMemory<uint16_t> sse_local_volume_d2(
        this->sse_cost_volume.size(), 0);
#else
    int const num_steps = this->cost_volume->channels();
    this->sgm_volume = mve::RawImage::create(width, height, num_steps);
    this->sgm_volume->fill(0);

    mve::RawImage::Ptr local_volume = this->sgm_volume->duplicate();
    mve::RawImage::Ptr local_volume_diag1 = this->sgm_volume->duplicate();
    mve::RawImage::Ptr local_volume_diag2 = this->sgm_volume->duplicate();
#endif

    /* process left-to-right */
#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
    std::fill(sse_local_volume.begin(), sse_local_volume.end(), 0);
    for (int y = 0; y < height; ++y)
    {
        int const x = 0;
        int const base = (y * y_stride + x) * d_stride;
        this->copy_cost_and_add_to_sgm(&sse_local_volume, base);
    }
    for (int x = 1; x < width; ++x)
        for (int y = 0; y < height; ++y)
            this->fill_path_cost_sse((y * y_stride + x) * d_stride,
                (y * y_stride + x - 1) * d_stride, &sse_local_volume);
#else
    local_volume->fill(0);
    for (int y = 0; y < height; ++y)
        for (int i = 0; i < num_steps; ++i)
        {
            local_volume->at(0, y, i) = this->cost_volume->at(0, y, i);
            this->sgm_volume->at(0, y, i) += this->cost_volume->at(0, y, i);
        }
    for (int x = 1; x < width; ++x)
        for (int y = 0; y < height; ++y)
            this->fill_path_cost(x, y, x - 1, y, local_volume);
#endif

    /* process right-to-left */
#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
    std::fill(sse_local_volume.begin(), sse_local_volume.end(), 0);
    for (int y = 0; y < height; ++y)
    {
        int const x = width - 1;
        int const base = (y * y_stride + x) * d_stride;
        this->copy_cost_and_add_to_sgm(&sse_local_volume, base);
    }
    for (int x = width - 2; x >= 0; --x)
        for (int y = 0; y < height; ++y)
            this->fill_path_cost_sse((y * y_stride + x) * d_stride,
                (y * y_stride + x + 1) * d_stride, &sse_local_volume);
#else
    local_volume->fill(0);
    for (int y = 0; y < height; ++y)
        for (int i = 0; i < num_steps; ++i)
        {
            local_volume->at(width - 1, y, i) =
                this->cost_volume->at(width - 1, y, i);
            this->sgm_volume->at(width - 1, y, i) +=
                this->cost_volume->at(width - 1, y, i);
        }
    for (int x = width - 2; x >= 0; --x)
        for (int y = 0; y < height; ++y)
            this->fill_path_cost(x, y, x + 1, y, local_volume);
#endif

    /* process top-to-bottom */
#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
    std::fill(sse_local_volume.begin(), sse_local_volume.end(), 0);
    std::fill(sse_local_volume_d1.begin(), sse_local_volume_d1.end(), 0);
    std::fill(sse_local_volume_d2.begin(), sse_local_volume_d2.end(), 0);
    for (int x = 0; x < width; ++x)
    {
        int const y = 0;
        int const base = (y * y_stride + x) * d_stride;
        this->copy_cost_and_add_to_sgm(&sse_local_volume, base);
        this->copy_cost_and_add_to_sgm(&sse_local_volume_d1, base);
        this->copy_cost_and_add_to_sgm(&sse_local_volume_d2, base);
    }
    for (int y = 0; y < height; ++y)
    {
        int const x = 0;
        int const base = (y * y_stride + x) * d_stride;
        this->copy_cost_and_add_to_sgm(&sse_local_volume_d1, base);
    }
    for (int y = 0; y < height; ++y)
    {
        int const x = width - 1;
        int const base = (y * y_stride + x) * d_stride;
        this->copy_cost_and_add_to_sgm(&sse_local_volume_d1, base);
    }
    for (int y = 1; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            if (x > 0)
                this->fill_path_cost_sse((y * y_stride + x) * d_stride,
                    ((y - 1) * y_stride + x - 1) * d_stride, &sse_local_volume);
            if (x < width - 1)
                this->fill_path_cost_sse((y * y_stride + x) * d_stride,
                    ((y - 1) * y_stride + x + 1) * d_stride, &sse_local_volume);
            this->fill_path_cost_sse((y * y_stride + x) * d_stride,
                    ((y - 1) * y_stride + x) * d_stride, &sse_local_volume);
        }

#else
    local_volume->fill(0);
    local_volume_diag1->fill(0);
    local_volume_diag2->fill(0);
    for (int x = 0; x < width; ++x)
        for (int i = 0; i < num_steps; ++i)
        {
            local_volume->at(x, 0, i) = this->cost_volume->at(x, 0, i);
            local_volume_diag1->at(x, 0, i) = this->cost_volume->at(x, 0, i);
            local_volume_diag2->at(x, 0, i) = this->cost_volume->at(x, 0, i);
            this->sgm_volume->at(x, 0, i) += this->cost_volume->at(x, 0, i) * 3;
        }

    for (int y = 0; y < height; ++y)
        for (int i = 0; i < num_steps; ++i)
        {
            local_volume_diag1->at(0, y, i) = this->cost_volume->at(0, y, i);
            this->sgm_volume->at(0, y, i) += this->cost_volume->at(0, y, i);
        }
    for (int y = 0; y < height; ++y)
        for (int i = 0; i < num_steps; ++i)
        {
            local_volume_diag2->at(width - 1, y, i) =
                this->cost_volume->at(width - 1, y, i);
            this->sgm_volume->at(width - 1, y, i) +=
                this->cost_volume->at(width - 1, y, i);
        }

    for (int y = 1; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            if (x > 0)
                this->fill_path_cost(x, y, x - 1, y - 1, local_volume_diag1);
            if (x < width - 1)
                this->fill_path_cost(x, y, x + 1, y - 1, local_volume_diag2);

            this->fill_path_cost(x, y, x, y - 1, local_volume);
        }
#endif

    /* process bottom-to-top */
#if SMVS_ENABLE_SSE && defined(__SSE4_1__)
    std::fill(sse_local_volume.begin(), sse_local_volume.end(), 0);
    std::fill(sse_local_volume_d1.begin(), sse_local_volume_d1.end(), 0);
    std::fill(sse_local_volume_d2.begin(), sse_local_volume_d2.end(), 0);
    for (int x = 0; x < width; ++x)
    {
        int const y = height - 1;
        int const base = (y * y_stride + x) * d_stride;
        this->copy_cost_and_add_to_sgm(&sse_local_volume, base);
        this->copy_cost_and_add_to_sgm(&sse_local_volume_d1, base);
        this->copy_cost_and_add_to_sgm(&sse_local_volume_d2, base);
    }
    for (int y = 0; y < height; ++y)
    {
        int const x = 0;
        int const base = (y * y_stride + x) * d_stride;
        this->copy_cost_and_add_to_sgm(&sse_local_volume_d1, base);
    }
    for (int y = 0; y < height; ++y)
    {
        int const x = width - 1;
        int const base = (y * y_stride + x) * d_stride;
        this->copy_cost_and_add_to_sgm(&sse_local_volume_d1, base);
    }
    for (int y = height - 2; y >= 0; --y)
        for (int x = 0; x < width; ++x)
        {
            if (x > 0)
                this->fill_path_cost_sse((y * y_stride + x) * d_stride,
                    ((y + 1) * y_stride + x - 1) * d_stride, &sse_local_volume);
            if (x < width - 1)
                this->fill_path_cost_sse((y * y_stride + x) * d_stride,
                    ((y + 1) * y_stride + x + 1) * d_stride, &sse_local_volume);
            this->fill_path_cost_sse((y * y_stride + x) * d_stride,
                    ((y + 1) * y_stride + x) * d_stride, &sse_local_volume);
        }
#else
    local_volume->fill(0);
    local_volume_diag1->fill(0);
    local_volume_diag2->fill(0);
    for (int x = 0; x < width; ++x)
        for (int i = 0; i < num_steps; ++i)
        {
            local_volume->at(x, height - 1, i) +=
                this->cost_volume->at(x, height - 1, i);
            local_volume_diag1->at(x, height - 1, i) +=
                this->cost_volume->at(x, height - 1, i);
            local_volume_diag2->at(x, height - 1, i) +=
                this->cost_volume->at(x, height - 1, i);
            this->sgm_volume->at(x, height - 1, i) +=
                this->cost_volume->at(x, height - 1, i) * 3;
        }
    for (int y = 0; y < height; ++y)
        for (int i = 0; i < num_steps; ++i)
        {
            local_volume_diag1->at(0, y, i) += this->cost_volume->at(0, y, i);
            this->sgm_volume->at(0, y, i) += this->cost_volume->at(0, y, i);
        }
    for (int y = 0; y < height; ++y)
        for (int i = 0; i < num_steps; ++i)
        {
            local_volume_diag2->at(width - 1, y, i) +=
                this->cost_volume->at(width - 1, y, i);
            this->sgm_volume->at(width - 1, y, i) +=
                this->cost_volume->at(width - 1, y, i);
        }

    for (int y = height - 2; y >= 0; --y)
        for (int x = 0; x < width; ++x)
        {
            if (x > 0)
                this->fill_path_cost(x, y, x - 1, y + 1, local_volume_diag1);
            if (x < width - 1)
                this->fill_path_cost(x, y, x + 1, y + 1, local_volume_diag2);

            this->fill_path_cost(x, y, x, y + 1, local_volume);
        }
#endif
}

void
SGMStereo::fill_depth_range_for_view (mve::Bundle::ConstPtr bundle,
    StereoView::Ptr view, float * range)
{
    std::vector<float> depth_values;
    int const width = view->get_width();
    int const height = view->get_height();

    mve::CameraInfo const& cam = view->get_camera();
    int const view_id = view->get_view_id();
    math::Matrix3f rot(cam.rot);
    math::Vec3f trans(cam.trans);
    float flen = cam.flen;

    double const fwidth2 = static_cast<double>(width) / 2.0;
    double const fheight2 = static_cast<double>(height) / 2.0;
    double const fnorm = static_cast<double>(std::max(width, height));

    mve::Bundle::Features const& features = bundle->get_features();
    for (std::size_t j = 0; j < features.size(); ++j)
    {
        mve::Bundle::Feature3D const& feat = features[j];
        for (std::size_t k = 0; k < feat.refs.size(); ++k)
            if (feat.refs[k].view_id == view_id)
            {
                math::Vec3f fpos(feat.pos);
                math::Vec3f proj = rot * fpos + trans;
                float const depth = proj[2];
                proj[0] = proj[0] * flen / proj[2];
                proj[1] = proj[1] * flen / proj[2];
                float const ix = proj[0] * fnorm + fwidth2;
                float const iy = proj[1] * fnorm + fheight2;
                int x = std::floor(ix);
                int y = std::floor(iy);

                if (x >= 0 && x < width && y >= 0 && y < height)
                    depth_values.push_back(depth);
                break;
            }
    }
    std::sort(depth_values.begin(), depth_values.end());

    if(depth_values.size() < 2)
    {
        range[0] = 0.3;
        range[1] = 1.1;
    } else
    {
        range[0] = depth_values.front() * 0.8f;
        range[1] = depth_values[(depth_values.size() * 9) / 10] * 1.2;
    }
}

SMVS_NAMESPACE_END
