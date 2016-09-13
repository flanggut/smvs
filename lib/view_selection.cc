/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "view_selection.h"

SMVS_NAMESPACE_BEGIN

mve::Scene::ViewList
ViewSelection::get_neighbors_for_view (std::size_t const view) const
{
    if (this->bundle != nullptr)
        return this->bundle_based_selection(view);
    else
        return this->position_based_selection(view);
}

mve::Scene::ViewList
ViewSelection::bundle_based_selection (std::size_t const view) const
{
    mve::Scene::ViewList neighbors;
    mve::View::Ptr main_view = this->views[view];
    if (!main_view->has_image(this->opts.embedding))
        return neighbors;

    math::Matrix4f main_view_wtc;
    math::Matrix3f main_view_iproj;
    main_view->get_camera().fill_world_to_cam(*main_view_wtc);
    main_view->get_camera().fill_inverse_calibration(*main_view_iproj,
        main_view->get_image_proxy(this->opts.embedding)->width,
        main_view->get_image_proxy(this->opts.embedding)->height);

    mve::Bundle::Features const& features = this->bundle->get_features();

    /* Create list of features for main view */
    mve::Bundle::Features main_view_features;
    std::vector<float> main_view_footprints;
    for (std::size_t f = 0; f < features.size(); ++f)
        if (features[f].contains_view_id(main_view->get_id()))
        {
            main_view_features.push_back(features[f]);
            math::Vec3f pos(features[f].pos);
            main_view_footprints.push_back(main_view_wtc.mult(pos, 1)[2] *
                main_view_iproj[0]);
        }

    /* Collect common features in neighboring views */
    std::multimap<std::size_t, std::size_t,
        std::greater<std::size_t>> common_features_map;
    for (std::size_t i = 0; i < this->views.size(); ++i)
    {
        mve::View::Ptr v = this->views[i];
        if (i == view || v == nullptr || v->get_camera().flen == 0.0
            || !v->has_image(this->opts.embedding))
            continue;
        math::Matrix4f neighbor_view_wtc;
        math::Matrix3f neighbor_view_iproj;
        v->get_camera().fill_world_to_cam(*neighbor_view_wtc);
        v->get_camera().fill_inverse_calibration(*neighbor_view_iproj,
            v->get_image_proxy(this->opts.embedding)->width,
            v->get_image_proxy(this->opts.embedding)->height);

        std::size_t num_matches = 0;
        for (std::size_t f = 0; f < main_view_features.size(); ++f)
        {
            if (main_view_features[f].contains_view_id(v->get_id()))
            {
                math::Vec3f pos(main_view_features[f].pos);
                float neighbor_footprint = neighbor_view_wtc.mult(pos, 1)[2] *
                    neighbor_view_iproj[0];
                if (std::min(neighbor_footprint, main_view_footprints[f]) /
                    std::max(neighbor_footprint, main_view_footprints[f]) > 0.6)
                    num_matches++;
            }
        }
        common_features_map.insert(std::make_pair(num_matches, i));
    }

    /* Use views with most common features as neighbors */
    for (auto features_for_view : common_features_map)
    {
        if (features_for_view.first > 10)
            neighbors.push_back(this->views[features_for_view.second]);
        if (neighbors.size() >= this->opts.num_neighbors)
            return neighbors;
    }
    return neighbors;
}


mve::Scene::ViewList
ViewSelection::position_based_selection (std::size_t const view) const
{
    mve::Scene::ViewList neighbors;

    mve::View::ConstPtr main_view = views[view];
    mve::CameraInfo const& main_cam = main_view->get_camera();
    math::Vec3f main_cam_pos;
    math::Vec3f main_cam_dir;
    main_cam.fill_camera_pos(*main_cam_pos);
    main_cam.fill_viewing_direction(*main_cam_dir);
    math::Vec3f main_cam_up;
    main_cam_up[0] = main_cam.rot[2];
    main_cam_up[1] = main_cam.rot[5];
    main_cam_up[2] = main_cam.rot[8];

    std::map<float, std::size_t> distances;
    for (std::size_t i = 0; i < views.size(); ++i)
    {
        if (views[i] == nullptr || i == view)
            continue;

        mve::CameraInfo const& cam = views[i]->get_camera();
        if (cam.flen == 0.0)
            continue;
        math::Vec3f pos;
        math::Vec3f dir;
        cam.fill_camera_pos(*pos);
        cam.fill_viewing_direction(*dir);

        math::Vec3f sub_cam_up;
        sub_cam_up[0] = cam.rot[2];
        sub_cam_up[1] = cam.rot[5];
        sub_cam_up[2] = cam.rot[8];

        if (main_cam_up.dot(sub_cam_up) < 0)
            continue;

        if (main_cam_dir.dot(dir) < 0.65)
            continue;
        float dist = (main_cam_pos - pos).norm();
        distances[dist] = i;
    }
    for (auto d : distances)
        neighbors.push_back(views[d.second]);

    return neighbors;
}

SMVS_NAMESPACE_END
