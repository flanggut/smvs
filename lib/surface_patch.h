/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_SURFACE_PATCH_HEADER
#define SMVS_SURFACE_PATCH_HEADER

#include <vector>

#include "mve/image.h"

#include "bicubic_patch.h"
#include "defines.h"

SMVS_NAMESPACE_BEGIN

class SurfacePatch
{
public:
    typedef std::shared_ptr<SurfacePatch> Ptr;
    typedef std::shared_ptr<SurfacePatch const> ConstPtr;

public:
    static SurfacePatch::Ptr create (int x, int y, int size,
        std::vector<BicubicPatch::Node::Ptr> & nodes);

    void fill_depth_map (mve::FloatImage & image);
    void fill_normal_map (mve::FloatImage & image, float inv_flen);

    int get_x (void) const;
    int get_y (void) const;

    void fill_values_at_nodes (std::vector<math::Vec2d> * pixels,
        std::vector<double> * depths = nullptr,
        std::vector<math::Vec2d> * first_deriv = nullptr,
        std::vector<double> * second_deriv = nullptr);

    void fill_values_at_pixels (std::vector<math::Vec2d> * pixels,
        std::vector<double> * depths = nullptr,
        std::vector<math::Vec2d> * first_deriv = nullptr,
        std::vector<math::Vec3d> * second_deriv = nullptr,
        std::vector<std::size_t> * pids = nullptr,
        int subsample = 1);

    void reset_interpolation (void);

    /* DEBUG */
    std::vector<BicubicPatch::Node::Ptr> get_nodes (void);

private:
    SurfacePatch (void);
    SurfacePatch (int x, int y, int size,
        std::vector<BicubicPatch::Node::Ptr> const& nodes);

private:
    int pixel_x;
    int pixel_y;
    int size;
    BicubicPatch::Ptr patch;
    BicubicPatch::Node::Ptr n00;
    BicubicPatch::Node::Ptr n10;
    BicubicPatch::Node::Ptr n01;
    BicubicPatch::Node::Ptr n11;
};

/* ------------------------ Implementation ------------------------ */

inline SurfacePatch::Ptr
SurfacePatch::create (int x, int y, int size,
    std::vector<BicubicPatch::Node::Ptr> & nodes)
{
    return std::shared_ptr<SurfacePatch>(new SurfacePatch(x, y, size, nodes));
}

inline
SurfacePatch::SurfacePatch (int x, int y, int size,
    std::vector<BicubicPatch::Node::Ptr> const& nodes)
    : pixel_x(x), pixel_y(y), size(size), patch(nullptr)
    , n00(nodes[0]), n10(nodes[1]), n01(nodes[2]), n11(nodes[3])
{
}

inline int
SurfacePatch::get_x (void) const
{
    return this->pixel_x;
}

inline int
SurfacePatch::get_y (void) const
{
    return this->pixel_y;
}

inline void
SurfacePatch::reset_interpolation (void)
{
    this->patch = nullptr;
}

inline std::vector<BicubicPatch::Node::Ptr>
SurfacePatch::get_nodes (void)
{
    std::vector<BicubicPatch::Node::Ptr> nodes;
    nodes.push_back(this->n00);
    nodes.push_back(this->n10);
    nodes.push_back(this->n01);
    nodes.push_back(this->n11);
    return nodes;
}

SMVS_NAMESPACE_END

#endif /* SMVS_SURFACE_PATCH_HEADER */
