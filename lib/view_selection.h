/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_VIEW_SELECTION_HEADER
#define SMVS_VIEW_SELECTION_HEADER

#include "mve/scene.h"

#include "defines.h"

SMVS_NAMESPACE_BEGIN

class ViewSelection
{
public:
    struct Options
    {
        Options (void) = default;
        std::size_t num_neighbors = 6;
        std::string embedding = "undistorted";
    };

public:
    ViewSelection (Options const& opts, mve::Scene::ViewList const& views,
        mve::Bundle::ConstPtr bundle = nullptr);

    mve::Scene::ViewList get_neighbors_for_view(std::size_t const view) const;

private:
    mve::Scene::ViewList bundle_based_selection(std::size_t const view) const;
    mve::Scene::ViewList position_based_selection(std::size_t const view) const;
    mve::Scene::ViewList get_sorted_neighbors(std::size_t const view) const;

private:
    Options const& opts;
    mve::Scene::ViewList const& views;
    mve::Bundle::ConstPtr bundle;
};

inline
ViewSelection::ViewSelection (ViewSelection::Options const& opts,
    mve::Scene::ViewList const& views, mve::Bundle::ConstPtr bundle)
    : opts(opts), views(views), bundle(bundle)
{
}

SMVS_NAMESPACE_END

#endif /* SMVS_VIEW_SELECTION_HEADER */
