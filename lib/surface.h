/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_SURFACE_HEADER
#define SMVS_SURFACE_HEADER

#include <vector>

#include "mve/bundle.h"
#include "mve/view.h"

#include "bicubic_patch.h"
#include "surface_patch.h"
#include "stereo_view.h"
#include "defines.h"

SMVS_NAMESPACE_BEGIN

class Surface
{
public:
    typedef std::shared_ptr<Surface> Ptr;
    typedef BicubicPatch::Node Node;
    typedef SurfacePatch Patch;
    typedef std::vector<BicubicPatch::Node::Ptr> NodeList;
    typedef std::vector<SurfacePatch::Ptr> PatchList;


public:
    static Ptr create(mve::Bundle::ConstPtr bundle, StereoView::Ptr main_view,
        int scale, mve::FloatImage::ConstPtr init_depth = nullptr);

    static Ptr create_planar(double depth, int width, int height, int scale);

    /* Get Data */
    mve::FloatImage::Ptr get_depth_map (void);
    mve::FloatImage::Ptr get_normal_map (float inv_flen);
    int get_scale (void) const;
    PatchList const& get_patches (void) const;
    NodeList const& get_nodes (void) const;
    int get_patchsize (void) const;
    int get_num_nodes (void) const;

    /* Global Operations */
    int expand (void);
    void subdivide_patches (void);
    void update_nodes(std::vector<double> const& delta,
        std::vector<double> * depth_updates);
    void fill_patches_from_bundle(mve::Bundle::ConstPtr bundle,
        mve::CameraInfo const& cam, int view_id);
    void fill_patches_from_depth (void);

    void fill_node_coords(std::vector<math::Vec2d> * coords);
    void fill_node_ids_for_patch (std::size_t patch_id,
        std::size_t * node_ids);
    void fill_node_ids_for_patch (std::size_t idx, std::size_t idy,
        std::size_t * node_ids);
    void fill_nodes_for_ids (std::size_t const* node_ids, NodeList * nodes);
    void fill_nodes_for_patch (std::size_t patch_id, NodeList * nodes);
    void fill_nodes_for_patch (std::size_t idx, std::size_t idy,
        NodeList * nodes);
    void fill_node_neighbors(std::size_t node_id, NodeList * neighbors,
        std::vector<std::size_t> * neighbor_ids = nullptr);
    void fill_node_derivatives_for_pixel (int pixel_id, double * d00,
        double * d10, double * d01, double * d11) const;

    void check_swap_nodes(Node::Ptr & ptr_base, Node::Ptr & ptr_new);

    void delete_node (std::size_t node_id);
    void delete_patch (std::size_t patch_id);
    void remove_patches_without_nodes (void);
    void remove_isolated_patches (void);
    void remove_nodes_without_patch (void);

private:
    Surface (mve::Bundle::ConstPtr bundle, int view_id,
        mve::CameraInfo const& cam, int width, int height, int scale);

    Surface (double depth, int width, int height, int scale);

    Surface (mve::Bundle::ConstPtr bundle, StereoView::Ptr main_view,
        int scale, mve::FloatImage::ConstPtr init_depth);

    void initialize_depth_from_bundle(mve::Bundle::ConstPtr bundle,
        mve::CameraInfo const& cam, int view_id);

    int fill_holes (void);

    void initialize_planar (double depth);
    void initialize_node_from_depth (int idx, int idy);

    SurfacePatch::Ptr get_patch (std::size_t idx, std::size_t idy);
    Node::Ptr get_node (std::size_t idx, std::size_t idy);
    std::size_t get_node_id (std::size_t idx, std::size_t idy);
    
    void create_patch (std::size_t idx, std::size_t idy);
    void delete_patch (std::size_t idx, std::size_t idy);
    void create_patch_with_nodes (std::size_t idx, std::size_t idy);

private:
    mve::FloatImage::Ptr depth;
    mve::FloatImage::Ptr return_depth;
    int const pixel_width;
    int const pixel_height;
    int pixel_start_x;
    int pixel_start_y;
    unsigned int num_patches_x;
    unsigned int num_patches_y;
    int node_stride;
    int scale;
    int patchsize;
    bool changed_node;
    std::vector<BicubicPatch::Node::Ptr> nodes;
    std::vector<SurfacePatch::Ptr> patches;
};

/* ------------------------ Implementation ------------------------ */

inline
Surface::Ptr
Surface::create(mve::Bundle::ConstPtr bundle, StereoView::Ptr main_view,
    int scale, mve::FloatImage::ConstPtr init_depth)
{
    return Ptr(new Surface(bundle, main_view, scale, init_depth));
}

inline
Surface::Ptr
Surface::create_planar(double depth, int width, int height, int scale)
{
    return Ptr(new Surface(depth, width, height, scale));
}

inline
Surface::Surface (double depth, int width, int height, int scale)
    : pixel_width(width)
    , pixel_height(height)
    , scale(scale)
{
    this->initialize_planar(depth);
    this->return_depth = this->depth->duplicate();
}

inline Surface::PatchList const&
Surface::get_patches (void) const
{
    return this->patches;
}

inline Surface::NodeList const&
Surface::get_nodes (void) const
{
    return this->nodes;
}

inline int
Surface::get_scale (void) const
{
    return this->scale;
}

inline void
Surface::delete_patch (std::size_t patch_id)
{
    this->patches[patch_id].reset();
}

inline int
Surface::get_patchsize (void) const
{
    return this->patchsize;
}

inline int
Surface::get_num_nodes (void) const
{
    return static_cast<int>(this->nodes.size());
}

inline SurfacePatch::Ptr
Surface::get_patch (std::size_t idx, std::size_t idy)
{
    return this->patches[idy * this->num_patches_x + idx];
}

inline Surface::Node::Ptr
Surface::get_node (std::size_t idx, std::size_t idy)
{
    return this->nodes[this->get_node_id(idx, idy)];
}

inline std::size_t
Surface::get_node_id (std::size_t idx, std::size_t idy)
{
    return idy * this->node_stride + idx;
}

SMVS_NAMESPACE_END

#endif /* SMVS_SURFACE_HEADER */
