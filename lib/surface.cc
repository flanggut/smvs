/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>

#include "mve/depthmap.h"
#include "mve/image_io.h"

#include "surface.h"

SMVS_NAMESPACE_BEGIN

Surface::Surface (mve::Bundle::ConstPtr bundle, StereoView::Ptr main_view,
    int scale, mve::FloatImage::ConstPtr init_depth)
    : pixel_width(main_view->get_width())
    , pixel_height(main_view->get_height())
    , scale(scale)
{
    int const width = this->pixel_width;
    int const height = this->pixel_height;

    this->patchsize = 1 << this->scale;
    this->num_patches_x = (width - 2) / this->patchsize - 1;
    this->num_patches_y = (height - 2) / this->patchsize - 1;
    this->node_stride = this->num_patches_x + 1;

    this->patches.resize(num_patches_x * num_patches_y, nullptr);
    this->nodes.resize((num_patches_x + 1) * (num_patches_y + 1), nullptr);

    this->pixel_start_x = (width - num_patches_x * this->patchsize) / 2;
    this->pixel_start_y = (height - num_patches_y * this->patchsize) / 2;

    this->depth = mve::FloatImage::create(width, height, 1);
    this->depth->fill(0.0f);

    /* Initialize depth using sparse SfM points */
    if (init_depth == nullptr)
        this->initialize_depth_from_bundle(bundle, main_view->get_camera(),
             main_view->get_view_id());
    else
        for (int p = 0; p < this->depth->get_pixel_amount(); ++p)
            if (init_depth->at(p) > 0.0)
                this->depth->at(p) = init_depth->at(p);

    this->fill_patches_from_depth();
    this->return_depth = this->depth->duplicate();
}

void
Surface::initialize_planar (double depth)
{
    int const width = this->pixel_width;
    int const height = this->pixel_height;

    this->depth = mve::FloatImage::create(width, height, 1);
    this->patchsize = 1 << this->scale;
    this->num_patches_x = (width - 2) / this->patchsize;
    this->num_patches_y = (height - 2) / this->patchsize;
    this->node_stride = this->num_patches_x + 1;
    std::cout << "Num patches " << num_patches_x << " "
    << num_patches_y << std::endl;

    this->patches.resize(num_patches_x * num_patches_y, nullptr);
    this->nodes.resize((num_patches_x + 1) * (num_patches_y + 1), nullptr);

    this->pixel_start_x = (width - num_patches_x * this->patchsize) / 2;
    this->pixel_start_y = (height - num_patches_y * this->patchsize) / 2;

    /* Initialize depth values of nodes */
    for (int i = 0; i < (int)num_patches_x + 1; ++i)
        for (int j = 0; j < (int)num_patches_y + 1; ++j)
        {
            BicubicPatch::Node::Ptr node =  BicubicPatch::Node::create();
            node->f = depth;
            node->f = 5.26 + 0.61 * i;
            node->dx = 0.61;
            this->nodes[j * (num_patches_x + 1) + i] = node;
        }

    /* Initialize patches by filling holes */
    this->fill_holes();
}

void
Surface::initialize_depth_from_bundle(mve::Bundle::ConstPtr bundle,
    const mve::CameraInfo &cam, int view_id)
{
    int const width = this->pixel_width;
    int const height = this->pixel_height;

    this->depth = mve::FloatImage::create(width, height, 1);

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

                if (x >= 0 && x < this->depth->width() &&
                    y >= 0 && y < this->depth->height())
                    this->depth->at(x, y, 0) = depth;
                break;
            }
    }
}

void
Surface::fill_patches_from_bundle(mve::Bundle::ConstPtr bundle,
    const mve::CameraInfo &cam, int view_id)
{
    this->initialize_depth_from_bundle(bundle, cam, view_id);
    this->fill_patches_from_depth();
}

void
Surface::fill_patches_from_depth (void)
{
    /* Initialize depth values of nodes */
    for (int i = 0; i < (int)num_patches_x + 1; ++i)
        for (int j = 0; j < (int)num_patches_y + 1; ++j)
            this->initialize_node_from_depth(i, j);

    /* Initialize patches by filling holes */
    this->fill_holes();

    this->remove_nodes_without_patch();
}


mve::FloatImage::Ptr
Surface::get_depth_map (void)
{
    mve::FloatImage::Ptr dmap = this->return_depth;
    dmap->fill(0.0);
    for (std::size_t i = 0; i < num_patches_x; ++i)
        for (std::size_t j = 0; j < num_patches_y; ++j)
            if (this->get_patch (i, j) == nullptr)
                continue;
            else
                this->get_patch(i, j)->fill_depth_map(*dmap);

    return dmap;
}

mve::FloatImage::Ptr
Surface::get_normal_map (float inv_flen)
{
    mve::FloatImage::Ptr normals =  mve::FloatImage::create(
        this->depth->width(), this->depth->height(), 3);
    for (std::size_t i = 0; i < num_patches_x; ++i)
        for (std::size_t j = 0; j < num_patches_y; ++j)
            if (this->get_patch (i, j) == nullptr)
                continue;
            else
                this->get_patch(i, j)->fill_normal_map(*normals, inv_flen);
    
    return normals;
}

void
Surface::create_patch(std::size_t idx, std::size_t idy)
{
    int const pixel_x = static_cast<int>(
        this->pixel_start_x + idx * this->patchsize);
    int const pixel_y = static_cast<int>(
        this->pixel_start_y + idy * this->patchsize);

    NodeList patch_nodes;
    this->fill_nodes_for_patch(idx, idy, &patch_nodes);
    this->patches[idy * this->num_patches_x + idx] =
        SurfacePatch::create(pixel_x, pixel_y, this->patchsize, patch_nodes);
}

void
Surface::delete_patch(std::size_t idx, std::size_t idy)
{
    this->patches[idy * this->num_patches_x + idx].reset();
}

void
Surface::create_patch_with_nodes(std::size_t idx, std::size_t idy)
{
    /* create all missing nodes */
    NodeList patch_nodes;
    this->fill_nodes_for_patch(idx, idy, &patch_nodes);
    if (patch_nodes[0] == nullptr)
        this->nodes[idy * node_stride + idx] =
            BicubicPatch::Node::create();
    if (patch_nodes[1] == nullptr)
        this->nodes[idy * node_stride + idx + 1] =
            BicubicPatch::Node::create();
    if (patch_nodes[2] == nullptr)
        this->nodes[(idy + 1) * node_stride + idx] =
            BicubicPatch::Node::create();
    if (patch_nodes[3] == nullptr)
        this->nodes[(idy + 1) * node_stride + idx + 1] =
            BicubicPatch::Node::create();

    /* create patch */
    this->create_patch(idx, idy);
}

void
Surface::delete_node (std::size_t node_id)
{
    this->nodes[node_id].reset();

    std::size_t idx = node_id % node_stride;
    std::size_t idy = node_id / node_stride;

    std::size_t patch_ids[4];
    patch_ids[3] = idy * this->num_patches_x + idx;
    patch_ids[2] = idy * this->num_patches_x + idx - 1;
    patch_ids[1] = (idy - 1) * this->num_patches_x + idx;
    patch_ids[0] = (idy - 1) * this->num_patches_x + idx - 1;

    if (idx == 0 && idy == 0)
        this->patches[patch_ids[3]].reset();
    else if (idx == 0)
    {
        if (idy < this->num_patches_y)
            this->patches[patch_ids[3]].reset();
        this->patches[patch_ids[1]].reset();
    }
    else if (idy == 0)
    {
        if (idx < this->num_patches_x)
            this->patches[patch_ids[3]].reset();
        this->patches[patch_ids[2]].reset();
    }
    else if (idx == this->num_patches_x && idy == this->num_patches_y)
    {
        this->patches[patch_ids[0]].reset();
    }
    else if (idx == this->num_patches_x)
    {
        if (idy > 0)
            this->patches[patch_ids[0]].reset();
        this->patches[patch_ids[2]].reset();
    }
    else if (idy == this->num_patches_y)
    {
        if (idx > 0)
            this->patches[patch_ids[0]].reset();
        this->patches[patch_ids[1]].reset();
    }
    else
    {
        this->patches[patch_ids[0]].reset();
        this->patches[patch_ids[1]].reset();
        this->patches[patch_ids[2]].reset();
        this->patches[patch_ids[3]].reset();
    }

}

void
Surface::fill_node_ids_for_patch (std::size_t patch_id, std::size_t * node_ids)
{
    std::size_t const idx = patch_id % this->num_patches_x;
    std::size_t const idy = patch_id / this->num_patches_x;
    this->fill_node_ids_for_patch(idx, idy, node_ids);
}

void
Surface::fill_node_ids_for_patch (std::size_t idx, std::size_t idy,
    std::size_t * node_ids)
{
    node_ids[0] = idy * node_stride + idx;
    node_ids[1] = idy * node_stride + idx + 1;
    node_ids[2] = (idy + 1) * node_stride + idx;
    node_ids[3] = (idy + 1) * node_stride + idx + 1;
}

void
Surface::fill_nodes_for_ids (std::size_t const* node_ids, NodeList * nodes)
{
    nodes->clear();
    nodes->push_back(this->nodes[node_ids[0]]);
    nodes->push_back(this->nodes[node_ids[1]]);
    nodes->push_back(this->nodes[node_ids[2]]);
    nodes->push_back(this->nodes[node_ids[3]]);
}

void
Surface::fill_nodes_for_patch (std::size_t patch_id, NodeList *nodes)
{
    std::size_t node_ids[4];
    this->fill_node_ids_for_patch(patch_id, node_ids);
    this->fill_nodes_for_ids(node_ids, nodes);
}

void
Surface::fill_nodes_for_patch (std::size_t idx, std::size_t idy,
    NodeList * nodes)
{
    std::size_t node_ids[4];
    this->fill_node_ids_for_patch(idx, idy, node_ids);
    this->fill_nodes_for_ids(node_ids, nodes);
}

void
Surface::fill_node_neighbors(std::size_t node_id, NodeList *neighbors,
    std::vector<std::size_t> * neighbor_ids)
{
    neighbors->clear();
    neighbors->resize(8);
    std::size_t idx = node_id % node_stride;
    std::size_t idy = node_id / node_stride;

    if (idx == 0 && idy == 0)
    {
        neighbors->at(4) = this->get_node(idx + 1, idy);
        neighbors->at(6) = this->get_node(idx, idy + 1);
        neighbors->at(7) = this->get_node(idx + 1, idy + 1);
    }
    else if (idx == 0)
    {
        neighbors->at(1) = this->get_node(idx, idy - 1);
        neighbors->at(2) = this->get_node(idx + 1, idy - 1);
        neighbors->at(4) = this->get_node(idx + 1, idy);
        if (idy < this->num_patches_y)
        {
            neighbors->at(6) = this->get_node(idx, idy + 1);
            neighbors->at(7) = this->get_node(idx + 1, idy + 1);
        }
    }
    else if (idy == 0)
    {
        neighbors->at(3) = this->get_node(idx - 1, idy);
        if (idx < this->num_patches_x)
            neighbors->at(4) = this->get_node(idx + 1, idy);
        neighbors->at(5) = this->get_node(idx - 1, idy + 1);
        neighbors->at(6) = this->get_node(idx, idy + 1);
        if (idx < this->num_patches_x)
            neighbors->at(7) = this->get_node(idx + 1, idy + 1);
    }
    else if (idx == this->num_patches_x && idy == this->num_patches_y)
    {
        neighbors->at(0) = this->get_node(idx - 1, idy -1);
        neighbors->at(1) = this->get_node(idx, idy - 1);
        neighbors->at(3) = this->get_node(idx - 1, idy);
    }
    else if (idx == this->num_patches_x)
    {
        if (idy > 0)
        {
            neighbors->at(0) = this->get_node(idx - 1, idy -1);
            neighbors->at(1) = this->get_node(idx, idy - 1);
        }
        neighbors->at(3) = this->get_node(idx - 1, idy);
        neighbors->at(5) = this->get_node(idx - 1, idy + 1);
        neighbors->at(6) = this->get_node(idx, idy + 1);
    }
    else if (idy == this->num_patches_y)
    {
        if (idx > 0)
            neighbors->at(0) = this->get_node(idx - 1, idy -1);
        neighbors->at(1) = this->get_node(idx, idy - 1);
        neighbors->at(2) = this->get_node(idx + 1, idy - 1);
        if (idx > 0)
            neighbors->at(3) = this->get_node(idx - 1, idy);
        neighbors->at(4) = this->get_node(idx + 1, idy);
    }
    else
    {
        neighbors->at(0) = this->get_node(idx - 1, idy -1);
        neighbors->at(1) = this->get_node(idx, idy - 1);
        neighbors->at(2) = this->get_node(idx + 1, idy - 1);
        neighbors->at(3) = this->get_node(idx - 1, idy);
        neighbors->at(4) = this->get_node(idx + 1, idy);
        neighbors->at(5) = this->get_node(idx - 1, idy + 1);
        neighbors->at(6) = this->get_node(idx, idy + 1);
        neighbors->at(7) = this->get_node(idx + 1, idy + 1);
    }

    if (neighbor_ids != nullptr)
    {
        if (idx == 0 && idy == 0)
        {
            neighbor_ids->push_back(this->get_node_id(idx + 1, idy));
            neighbor_ids->push_back(this->get_node_id(idx, idy + 1));
            neighbor_ids->push_back(this->get_node_id(idx + 1, idy + 1));
        }
        else if (idx == 0)
        {
            neighbor_ids->push_back(this->get_node_id(idx, idy - 1));
            neighbor_ids->push_back(this->get_node_id(idx + 1, idy - 1));
            neighbor_ids->push_back(this->get_node_id(idx + 1, idy));
            if (idy < this->num_patches_y)
            {
                neighbor_ids->push_back(this->get_node_id(idx, idy + 1));
                neighbor_ids->push_back(this->get_node_id(idx + 1, idy + 1));
            }
        }
        else if (idy == 0)
        {
            neighbor_ids->push_back(this->get_node_id(idx - 1, idy));
            if (idx < this->num_patches_x)
                neighbor_ids->push_back(this->get_node_id(idx + 1, idy));
            neighbor_ids->push_back(this->get_node_id(idx - 1, idy + 1));
            neighbor_ids->push_back(this->get_node_id(idx, idy + 1));
            if (idx < this->num_patches_x)
                neighbor_ids->push_back(this->get_node_id(idx + 1, idy + 1));
        }
        else if (idx == this->num_patches_x && idy == this->num_patches_y)
        {
            neighbor_ids->push_back(this->get_node_id(idx - 1, idy -1));
            neighbor_ids->push_back(this->get_node_id(idx, idy - 1));
            neighbor_ids->push_back(this->get_node_id(idx - 1, idy));
        }
        else if (idx == this->num_patches_x)
        {
            if (idy > 0)
            {
                neighbor_ids->push_back(this->get_node_id(idx - 1, idy -1));
                neighbor_ids->push_back(this->get_node_id(idx, idy - 1));
            }
            neighbor_ids->push_back(this->get_node_id(idx - 1, idy));
            neighbor_ids->push_back(this->get_node_id(idx - 1, idy + 1));
            neighbor_ids->push_back(this->get_node_id(idx, idy + 1));
        }
        else if (idy == this->num_patches_y)
        {
            if (idx > 0)
                neighbor_ids->push_back(this->get_node_id(idx - 1, idy -1));
            neighbor_ids->push_back(this->get_node_id(idx, idy - 1));
            neighbor_ids->push_back(this->get_node_id(idx + 1, idy - 1));
            if (idx > 0)
                neighbor_ids->push_back(this->get_node_id(idx - 1, idy));
            neighbor_ids->push_back(this->get_node_id(idx + 1, idy));
        }
        else
        {
            neighbor_ids->push_back(this->get_node_id(idx - 1, idy -1));
            neighbor_ids->push_back(this->get_node_id(idx, idy - 1));
            neighbor_ids->push_back(this->get_node_id(idx + 1, idy - 1));
            neighbor_ids->push_back(this->get_node_id(idx - 1, idy));
            neighbor_ids->push_back(this->get_node_id(idx + 1, idy));
            neighbor_ids->push_back(this->get_node_id(idx - 1, idy + 1));
            neighbor_ids->push_back(this->get_node_id(idx, idy + 1));
            neighbor_ids->push_back(this->get_node_id(idx + 1, idy + 1));
        }
    }
}

void
Surface::check_swap_nodes(Node::Ptr & ptr_base, Node::Ptr & ptr_new)
{
    if (ptr_base == nullptr || ptr_new->f * 0.9 > ptr_base->f)
    {
        std::swap(ptr_base, ptr_new);
        this->changed_node = true;
    }
}

int
Surface::expand (void)
{
    std::map<std::size_t, Node::Ptr> new_nodes;

    for (int iter = 0; iter < 2; ++iter)
    {
    /* Step 1: Expand Surface at the borders */
    for (std::size_t node_id = 0; node_id < this->nodes.size(); ++node_id)
    {
        NodeList neighbors;
        std::vector<std::size_t> neighbor_ids;
        this->fill_node_neighbors(node_id, &neighbors, &neighbor_ids);

//        this->initialize_node_from_depth(node_id % this->node_stride,
//            static_cast<int>(node_id) / this->node_stride);

        if (this->nodes[node_id] != nullptr
            && new_nodes[node_id] == nullptr)
            continue;

        Node::Ptr node;

        bool top_left = (neighbors[0] != nullptr && neighbors[1] != nullptr
            && neighbors[3] != nullptr);
        bool top_right = (neighbors[1] != nullptr && neighbors[2] != nullptr
            && neighbors[4] != nullptr);
        bool bottom_left = (neighbors[3] != nullptr && neighbors[5] != nullptr
            && neighbors[6] != nullptr);
        bool bottom_right = (neighbors[4] != nullptr && neighbors[6] != nullptr
            && neighbors[7] != nullptr);

        bool top =  (neighbors[0] != nullptr && neighbors[1] != nullptr
            && neighbors[2] != nullptr);
        bool left = (neighbors[0] != nullptr && neighbors[3] != nullptr
            && neighbors[5] != nullptr);
        bool bottom = (neighbors[5] != nullptr && neighbors[6] != nullptr
            && neighbors[7] != nullptr);
        bool right = (neighbors[2] != nullptr && neighbors[4] != nullptr
            && neighbors[7] != nullptr);

        /* top left */
        if (top_left)
        {
            node = Node::create();
            node->f = ((neighbors[3]->f + neighbors[3]->dx / 2.0) +
                (neighbors[1]->f + neighbors[1]->dy / 2.0)) / 2.0;
//            node->dx = neighbors[3]->dx;
//            node->dy = neighbors[1]->dy;
//            node->dxy = neighbors[0]->dxy;
            check_swap_nodes(new_nodes[node_id], node);
        }
        /* top right */
        if (top_right)
        {
            node = Node::create();
            node->f = ((neighbors[4]->f - neighbors[4]->dx / 2.0) +
                       (neighbors[1]->f + neighbors[1]->dy / 2.0)) / 2.0;
//            node->dx = neighbors[4]->dx;
//            node->dy = neighbors[1]->dy;
//            node->dxy = neighbors[2]->dxy;
            check_swap_nodes(new_nodes[node_id], node);
        }
        /* bottom left */
        if (bottom_left)
        {
            node = Node::create();
            node->f = ((neighbors[3]->f + neighbors[3]->dx / 2.0) +
                       (neighbors[6]->f - neighbors[6]->dy / 2.0)) / 2.0;
//            node->dx = neighbors[3]->dx;
//            node->dy = neighbors[6]->dy;
//            node->dxy = neighbors[5]->dxy;
            check_swap_nodes(new_nodes[node_id], node);
        }
        /* bottom right */
        if (bottom_right)
        {
            node = Node::create();
            node->f = ((neighbors[4]->f - neighbors[4]->dx / 2.0) +
                       (neighbors[6]->f - neighbors[6]->dy / 2.0)) / 2.0;
//            node->dx = neighbors[4]->dx;
//            node->dy = neighbors[6]->dy;
//            node->dxy = neighbors[7]->dxy;
            check_swap_nodes(new_nodes[node_id], node);
        }
        /* top */
        if (top)
        {
            node = Node::create();
            node->f = ((neighbors[0]->f + neighbors[0]->dy / 2.0) +
                       (neighbors[1]->f + neighbors[1]->dy / 2.0) +
                       (neighbors[2]->f + neighbors[2]->dy / 2.0)) / 3.0;
//            node->dx = neighbors[1]->dx;
//            node->dy = neighbors[1]->dy;
//            node->dxy = neighbors[1]->dxy;
            check_swap_nodes(new_nodes[node_id], node);
        }
        /* left */
        if (left)
        {
            node = Node::create();
            node->f = ((neighbors[0]->f + neighbors[0]->dx / 2.0) +
                       (neighbors[3]->f + neighbors[3]->dx / 2.0) +
                       (neighbors[5]->f + neighbors[5]->dx / 2.0)) / 3.0;
//            node->dx = neighbors[3]->dx;
//            node->dy = neighbors[3]->dy;
//            node->dxy = neighbors[3]->dxy;
            check_swap_nodes(new_nodes[node_id], node);
        }
        /* bottom */
        if (bottom)
        {
            node = Node::create();
            node->f = ((neighbors[5]->f - neighbors[5]->dy / 2.0) +
                       (neighbors[6]->f - neighbors[6]->dy / 2.0) +
                       (neighbors[7]->f - neighbors[7]->dy / 2.0)) / 3.0;
//            node->dx = neighbors[6]->dx;
//            node->dy = neighbors[6]->dy;
//            node->dxy = neighbors[6]->dxy;
            check_swap_nodes(new_nodes[node_id], node);
        }
        /* right */
        if (right)
        {
            node = Node::create();
            node->f = ((neighbors[2]->f - neighbors[2]->dx / 2.0) +
                       (neighbors[4]->f - neighbors[4]->dx / 2.0) +
                       (neighbors[7]->f - neighbors[7]->dx / 2.0)) / 3.0;
//            node->dx = neighbors[4]->dx;
//            node->dy = neighbors[4]->dy;
//            node->dxy = neighbors[4]->dxy;
            check_swap_nodes(new_nodes[node_id], node);
        }
    }
    for (auto n : new_nodes)
        if (n.second != nullptr)
            this->nodes[n.first] = n.second;
    }

    /* Step 2: Fill remaining holes */
    int filled_count = this->fill_holes();

    /* Step 3: Clear remaining nodes */
    this->remove_nodes_without_patch();

    return filled_count;
}

int
Surface::fill_holes (void)
{
    int filled_counter = 0;
    for (std::size_t x = 0; x < num_patches_x; ++x)
        for (std::size_t y = 0; y < num_patches_y; ++y)
        {
            if (this->get_patch(x, y) != nullptr)
                continue;
            NodeList patch_nodes;
            this->fill_nodes_for_patch(x, y, &patch_nodes);
            if (patch_nodes[0] != nullptr &&
                patch_nodes[1] != nullptr &&
                patch_nodes[2] != nullptr &&
                patch_nodes[3] != nullptr)
            {
                this->create_patch(x, y);
                filled_counter += 1;
            }
        }
    return filled_counter;
}

void
Surface::fill_node_coords(std::vector<math::Vec2d> * coords)
{
    coords->clear();
    for (std::size_t i = 0; i < this->nodes.size(); ++i)
    {
        std::size_t idx = i % this->node_stride;
        std::size_t idy = i / this->node_stride;
        std::size_t const x = idx * this->patchsize + this->pixel_start_x;
        std::size_t const y = idy * this->patchsize + this->pixel_start_y;
        coords->emplace_back(x, y);
    }
}

void
Surface::initialize_node_from_depth (int idx, int idy)
{
    int const x = idx * this->patchsize + this->pixel_start_x;
    int const y = idy * this->patchsize + this->pixel_start_y;

    if (this->nodes[idy * node_stride + idx] != nullptr) return;

    int const window_size = this->patchsize / 2;

    std::vector<double> d[4];
    for (int i = -window_size; i < 0; ++i)
        for (int j = -window_size; j < 0; ++j)
            if (x + i >= 0 && x + i < this->depth->width() &&
                y + j >= 0 && y + j < this->depth->height() &&
                this->depth->at(x + i, y + j, 0) > 0.0)
                d[0].emplace_back(this->depth->at(x + i, y + j, 0));

    for (int i = 0; i < window_size; ++i)
        for (int j = -window_size; j < 0; ++j)
            if (x + i >= 0 && x + i < this->depth->width() &&
                y + j >= 0 && y + j < this->depth->height() &&
                this->depth->at(x + i, y + j, 0) > 0.0)
                d[1].emplace_back(this->depth->at(x + i, y + j, 0));

    for (int i = -window_size; i < 0; ++i)
        for (int j = 0; j < window_size; ++j)
            if (x + i >= 0 && x + i < this->depth->width() &&
                y + j >= 0 && y + j < this->depth->height() &&
                this->depth->at(x + i, y + j, 0) > 0.0)
                d[2].emplace_back(this->depth->at(x + i, y + j, 0));

    for (int i = 0; i < window_size; ++i)
        for (int j = 0; j < window_size; ++j)
            if (x + i >= 0 && x + i < this->depth->width() &&
                y + j >= 0 && y + j < this->depth->height() &&
                this->depth->at(x + i, y + j, 0) > 0.0)
                d[3].emplace_back(this->depth->at(x + i, y + j, 0));


    int num_non_zeros = 4;

    double avg[4];
    double sum = 0.0;
    std::vector<double> all;
    for (int i = 0; i < 4; ++i)
    {
        for (std::size_t j = 0; j < d[i].size(); ++j)
        {
            sum += d[i][j];
            all.push_back(d[i][j]);
        }

        if (d[i].size() == 0)
        {
            avg[i] = 0.0;
            num_non_zeros -= 1;
            continue;
        }
        avg[i] = *std::min_element(d[i].begin(), d[i].end());
    }

    if (num_non_zeros == 0)
        return;

    if (d[0].size() + d[1].size() + d[2].size() + d[3].size() < 2)
        return;

    sum /= d[0].size() + d[1].size() + d[2].size() + d[3].size();

    std::nth_element(all.begin(), all.begin() + all.size() / 2, all.end());

    BicubicPatch::Node::Ptr node =  BicubicPatch::Node::create();
    node->f = all[all.size() / 2];

    if (num_non_zeros == 4)
    {
        node->dx = ((avg[1] + avg[3]) - (avg[0] + avg[2])) / 2.0;
        node->dy = ((avg[2] + avg[3]) - (avg[0] + avg[1])) / 2.0;
        node->dxy = ((avg[3] - avg[2]) - (avg[1] - avg[0]));
    } else
    {
        if ((avg[1] == 0 || avg[0] == 0) && avg[3] != 0 && avg[2] != 0)
            node->dx = (avg[3] - avg[2]);
        else if ((avg[2] == 0 || avg[3] == 0) && avg[1] != 0 && avg[0] != 0)
            node->dx = (avg[1] - avg[0]);

        if ((avg[0] == 0 || avg[2] == 0) && avg[3] != 0 && avg[1] != 0)
            node->dy = (avg[3] - avg[1]);
        else if ((avg[1] == 0 || avg[2] == 0) && avg[0] != 0 && avg[2] != 0)
            node->dy = (avg[2] - avg[0]);
    }
    this->nodes[idy * node_stride + idx] = node;
}

void
Surface::remove_nodes_without_patch (void)
{
    int num_removed = 0;
    for (std::size_t i = 0; i < this->nodes.size(); ++i)
    {
        if (this->nodes[i] == nullptr)
            continue;

        std::size_t idx = i % node_stride;
        std::size_t idy = i / node_stride;

        std::size_t patch_ids[4];
        patch_ids[3] = idy * this->num_patches_x + idx;
        patch_ids[2] = idy * this->num_patches_x + idx - 1;
        patch_ids[1] = (idy - 1) * this->num_patches_x + idx;
        patch_ids[0] = (idy - 1) * this->num_patches_x + idx - 1;

        if (idx == 0 && idy == 0)
        {
            if (this->patches[patch_ids[3]] == nullptr)
            {
                this->nodes[i].reset();
                num_removed += 1;
            }
        }
        else if (idx == 0)
        {
            if (idy == this->num_patches_y)
            {
                if (this->patches[patch_ids[1]] == nullptr)
                {
                    this->nodes[i].reset();
                    num_removed += 1;
                }
            } else if (this->patches[patch_ids[1]] == nullptr
                && this->patches[patch_ids[3]] == nullptr)
            {
                this->nodes[i].reset();
                num_removed += 1;
            }
        }
        else if (idy == 0)
        {
            if (idx == this->num_patches_x)
            {
                if (this->patches[patch_ids[2]] == nullptr)
                {
                    this->nodes[i].reset();
                    num_removed += 1;
                }
            } else if( this->patches[patch_ids[3]] == nullptr
                && this->patches[patch_ids[2]] == nullptr)
            {
                this->nodes[i].reset();
                num_removed += 1;
            }
        }
        else if (idx == this->num_patches_x && idy == this->num_patches_y)
        {
            if (this->patches[patch_ids[0]] == nullptr)
            {
                this->nodes[i].reset();
                num_removed += 1;
            }
        }
        else if (idx == this->num_patches_x)
        {
            if (idy == 0)
            {
                if (this->patches[patch_ids[2]] == nullptr)
                {
                    this->nodes[i].reset();
                    num_removed += 1;
                }
            } else if (this->patches[patch_ids[0]] == nullptr
                && this->patches[patch_ids[2]] == nullptr)
            {
                this->nodes[i].reset();
                num_removed += 1;
            }
        }
        else if (idy == this->num_patches_y)
        {
            if (idx == 0)
            {
                if (this->patches[patch_ids[1]] == nullptr)
                {
                    this->nodes[i].reset();
                    num_removed += 1;
                }
            } else if (this->patches[patch_ids[0]] == nullptr
                && this->patches[patch_ids[1]] == nullptr)
            {
                this->nodes[i].reset();
                num_removed += 1;
            }
        }
        else if (this->patches[patch_ids[0]] == nullptr
            && this->patches[patch_ids[1]] == nullptr
            && this->patches[patch_ids[2]] == nullptr
            && this->patches[patch_ids[3]] == nullptr)
        {
            this->nodes[i].reset();
            num_removed += 1;
        }
    }
}

void
Surface::remove_patches_without_nodes (void)
{
    for (std::size_t i = 0; i < this->patches.size(); ++i)
    {
        std::size_t node_ids[4];
        this->fill_node_ids_for_patch(i, node_ids);
        for (int n = 0; n < 4; ++n)
            if (this->nodes[node_ids[n]] == nullptr)
            {
                this->delete_patch(i);
                break;
            }
    }
}

void
Surface::remove_isolated_patches (void)
{
    for (std::size_t x = 0; x < this->num_patches_x; ++x)
        for (std::size_t y = 0; y < this->num_patches_y; ++y)
        {
            if (this->get_patch(x, y) == nullptr)
                continue;
            int valid_neighbors = 0;
            if (x > 0 && y > 0 &&
                this->get_patch(x - 1, y - 1) != nullptr)
                valid_neighbors += 1;
            if (x > 0 &&
                this->get_patch(x - 1, y) != nullptr)
                valid_neighbors += 1;
            if (x > 0 && y < this->num_patches_y - 1 &&
                this->get_patch(x - 1, y + 1) != nullptr)
                valid_neighbors += 1;

            if (y > 0 &&
                this->get_patch(x, y - 1) != nullptr)
                valid_neighbors += 1;
            if (y < this->num_patches_y - 1 &&
                this->get_patch(x, y + 1) != nullptr)
                valid_neighbors += 1;

            if (x < this->num_patches_x - 1 && y > 0 &&
                this->get_patch(x + 1, y - 1) != nullptr)
                valid_neighbors += 1;
            if (x < this->num_patches_x - 1 &&
                this->get_patch(x + 1, y) != nullptr)
                valid_neighbors += 1;
            if (x < this->num_patches_x - 1 && y < this->num_patches_y - 1 &&
                this->get_patch(x + 1, y + 1) != nullptr)
                valid_neighbors += 1;

            if (valid_neighbors < 3)
                this->delete_patch(x, y);
        }
    this->remove_nodes_without_patch();
}

void
Surface::fill_node_derivatives_for_pixel (int pixel_id, double * d00,
    double * d10, double * d01, double * d11) const
{
    int const i = pixel_id % this->patchsize;
    int const j = pixel_id / this->patchsize;
    double x = (static_cast<double>(i) + 0.5) / this->patchsize;
    double y = (static_cast<double>(j) + 0.5) / this->patchsize;
    BicubicPatch::node_derivatives(x, y, d00, d10, d01, d11);

    double patch_to_pixel = 1.0 /  this->patchsize;
    
    for (int i = 4; i < 24; ++i)
    {
        d00[i] *= patch_to_pixel;
        d10[i] *= patch_to_pixel;
        d01[i] *= patch_to_pixel;
        d11[i] *= patch_to_pixel;
    }
    for (int i = 12; i < 24; ++i)
    {
        d00[i] *= patch_to_pixel;
        d10[i] *= patch_to_pixel;
        d01[i] *= patch_to_pixel;
        d11[i] *= patch_to_pixel;
    }
}

void
Surface::update_nodes (std::vector<double> const& delta,
    std::vector<double> * updates)
{
    updates->resize(delta.size() / 4);

    /* add delta to patches */
    for (std::size_t i = 0; i < this->nodes.size(); ++i)
    {
        if (this->nodes[i] == nullptr)
                continue;

        updates->at(i) = delta[i * 4 + 0] / this->nodes[i]->f;

        this->nodes[i]->f   += delta[i * 4 + 0];
        this->nodes[i]->dx  += delta[i * 4 + 1];
        this->nodes[i]->dy  += delta[i * 4 + 2];
        this->nodes[i]->dxy += delta[i * 4 + 3];
    }

    /* reset patch interpolations */
    for (std::size_t i = 0; i < this->patches.size(); ++i)
        if (this->patches[i] != nullptr)
            this->patches[i]->reset_interpolation();
}

void
Surface::subdivide_patches (void)
{
    this->scale = this->scale - 1;
    this->patchsize = 1 << this->scale;

    int new_num_patches_x = (this->pixel_width - 2) / this->patchsize;
    int new_num_patches_y = (this->pixel_height - 2) / this->patchsize;
    int offset_x = new_num_patches_x - (this->num_patches_x * 2);
    int offset_y = new_num_patches_y - (this->num_patches_y * 2);
    if (offset_x >= 2)
    {
        new_num_patches_x = this->num_patches_x * 2 + 2;
        this->pixel_start_x = (this->pixel_width -
            new_num_patches_x * this->patchsize) / 2;
        offset_x = 1;
    } else
    {
        offset_x = 0;
        new_num_patches_x = this->num_patches_x * 2;
    }
    if (offset_y >= 2)
    {
        new_num_patches_y = this->num_patches_y * 2 + 2;
        this->pixel_start_y = (this->pixel_height -
            new_num_patches_y * this->patchsize) / 2;
        offset_y = 1;
    } else
    {
        offset_y = 0;
        new_num_patches_y = this->num_patches_y * 2;
    }

    NodeList new_nodes;
    new_nodes.resize((new_num_patches_x + 1)
        * (new_num_patches_y + 1), nullptr);
    int const new_node_stride = new_num_patches_x + 1;

    /* create 5 new nodes for each patch */
    for (std::size_t patch_id = 0; patch_id < this->patches.size(); ++patch_id)
    {
        if (this->patches[patch_id] == nullptr)
            continue;

        std::size_t const idx = (int)patch_id % this->num_patches_x;
        std::size_t const idy = (int)patch_id / this->num_patches_x;
        std::size_t const new_idx = 2 * idx + offset_x;
        std::size_t const new_idy = 2 * idy + offset_y;

        NodeList old_nodes;
        this->fill_nodes_for_patch(patch_id, &old_nodes);
        BicubicPatch::Ptr patch = BicubicPatch::create(old_nodes[0],
            old_nodes[1], old_nodes[2], old_nodes[3]);

        BicubicPatch::Node::Ptr new_node;

        new_node = BicubicPatch::Node::create();
        new_node->f = patch->evaluate_f(0.5, 0.0);
        new_node->dx = patch->evaluate_dx(0.5, 0.0) / 2;
        new_node->dy = patch->evaluate_dy(0.5, 0.0) / 2;
        new_node->dxy = patch->evaluate_dxy(0.5, 0.0) / 4;
        new_nodes[new_idx + 1 + new_node_stride * (new_idy)] = new_node;

        new_node = BicubicPatch::Node::create();
        new_node->f = patch->evaluate_f(0.0, 0.5);
        new_node->dx = patch->evaluate_dx(0.0, 0.5) / 2;
        new_node->dy = patch->evaluate_dy(0.0, 0.5) / 2;
        new_node->dxy = patch->evaluate_dxy(0.0, 0.5) / 4;
        new_nodes[new_idx + new_node_stride * (new_idy +1)] = new_node;

        new_node = BicubicPatch::Node::create();
        new_node->f = patch->evaluate_f(0.5, 0.5);
        new_node->dx = patch->evaluate_dx(0.5, 0.5) / 2;
        new_node->dy = patch->evaluate_dy(0.5, 0.5) / 2;
        new_node->dxy = patch->evaluate_dxy(0.5, 0.5) / 4;
        new_nodes[new_idx + 1 + new_node_stride * (new_idy +1)] = new_node;

        new_node = BicubicPatch::Node::create();
        new_node->f = patch->evaluate_f(1.0, 0.5);
        new_node->dx = patch->evaluate_dx(1.0, 0.5) / 2;
        new_node->dy = patch->evaluate_dy(1.0, 0.5) / 2;
        new_node->dxy = patch->evaluate_dxy(1.0, 0.5) / 4;
        new_nodes[new_idx + 2 + new_node_stride * (new_idy +1)] = new_node;

        new_node = BicubicPatch::Node::create();
        new_node->f = patch->evaluate_f(0.5, 1.0);
        new_node->dx = patch->evaluate_dx(0.5, 1.0) / 2;
        new_node->dy = patch->evaluate_dy(0.5, 1.0) / 2;
        new_node->dxy = patch->evaluate_dxy(0.5, 1.0) / 4;
        new_nodes[new_idx + 1 + new_node_stride * (new_idy + 2)] = new_node;
    }

    /* copy old nodes */
    for (std::size_t node_id = 0; node_id < this->nodes.size(); ++node_id)
    {
        if (this->nodes[node_id] == nullptr)
            continue;

        std::size_t const idx = node_id % node_stride;
        std::size_t const idy = node_id / node_stride;
        std::size_t const new_idx = 2 * idx + offset_x;
        std::size_t const new_idy = 2 * idy + offset_y;
        std::size_t const new_node_id = new_idx + (new_node_stride * new_idy);

        this->nodes[node_id]->dx /= 2;
        this->nodes[node_id]->dy /= 2;
        this->nodes[node_id]->dxy /= 4;

        new_nodes[new_node_id] = this->nodes[node_id];
    }

    /* move new data */
    this->num_patches_x = new_num_patches_x;
    this->num_patches_y = new_num_patches_y;
    this->node_stride = this->num_patches_x + 1;
    this->nodes = new_nodes;

    /* create new patches */
    this->patches.clear();
    this->patches.resize(this->num_patches_x * this->num_patches_y);
    this->fill_holes();

    /* clear remaining nodes */
    this->remove_nodes_without_patch();
}

SMVS_NAMESPACE_END
