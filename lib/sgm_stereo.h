/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_SGM_STEREO_HEADER
#define SMVS_SGM_STEREO_HEADER

#include "mve/image.h"
#include "util/aligned_memory.h"

#include "stereo_view.h"
#include "defines.h"

SMVS_NAMESPACE_BEGIN

class SGMStereo
{
public:
    struct Options
    {
        Options (void) = default;
        int debug_lvl = 0;
        int scale = 1;
        int num_steps = 64;
        float min_depth = 0.0f;
        float max_depth = 0.0f;
        uint16_t penalty1 = 24;
        uint16_t penalty2 = 4000;
    };

    SGMStereo (Options const& opts,
       StereoView::Ptr main, StereoView::Ptr neighbor);

    static mve::FloatImage::Ptr reconstruct (SGMStereo::Options sgm_opts,
        StereoView::Ptr main_view, StereoView::Ptr neighbor,
        mve::Bundle::ConstPtr bundle = nullptr);

    mve::FloatImage::Ptr run_sgm (float min_depth, float max_depth);

private:
    void warped_neighbor_for_depth (float depth, mve::ByteImage::Ptr image);

    void census_filter (mve::ByteImage::ConstPtr image,
        mve::Image<uint64_t>::Ptr filtered);

    void create_cost_volume (float min_depth, float max_depth, int num_steps);

    void aggregate_sgm_costs (void);

    void fill_path_cost (int x, int y, int px, int py,
        mve::RawImage::Ptr path);

    void fill_path_cost_sse (int base, int base_prev,
        util::AlignedMemory<uint16_t> * path);
    void copy_cost_and_add_to_sgm (util::AlignedMemory<uint16_t> * local_volume,
        int base);
    uint16_t sse_reduction_min (uint16_t * data, std::size_t size);

    mve::FloatImage::Ptr depth_from_cost_volume (void);
    mve::FloatImage::Ptr depth_from_sgm_volume (void);

    static void fill_depth_range_for_view (mve::Bundle::ConstPtr bundle,
        StereoView::Ptr view, float * range);

private:
    Options opts;

    StereoView::Ptr main;
    StereoView::Ptr neighbor;
    mve::ByteImage::ConstPtr main_image;
    mve::ByteImage::ConstPtr neighbor_image;

    mve::ByteImage::Ptr cost_volume;
    mve::RawImage::Ptr sgm_volume;
    std::vector<float> cost_volume_depths;

    util::AlignedMemory<uint16_t> sse_cost_volume;
    util::AlignedMemory<uint16_t> sse_sgm_volume;
    util::AlignedMemory<uint16_t> min_cost_updates;
    util::AlignedMemory<uint16_t> cost_updates;
    util::AlignedMemory<uint16_t> mins;
};

SMVS_NAMESPACE_END

#endif /* SMVS_SGM_STEREO_HEADER */
