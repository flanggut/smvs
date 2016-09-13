/*
 * Copyright (C) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "mve/scene.h"
#include "mve/view.h"
#include "mve/camera.h"
#include "mve/depthmap.h"

#include "util/file_system.h"
#include "util/timer.h"

#include "correspondence.h"
#include "depth_optimizer.h"
#include "surface.h"

int main (int argc, char** argv)
{
    /* initialize scene */
    std::string base = "/tmp/testscene";
    std::string base_views = "/tmp/testscene/views";
    util::fs::mkdir(base.c_str());
    util::fs::mkdir(base_views.c_str());

    mve::Scene::Ptr scene = mve::Scene::create("/tmp/testscene");

    mve::View::Ptr view0 = mve::View::create();
    view0->set_id(0);
    view0->set_name("000");
    mve::View::Ptr view1 = mve::View::create();
    view1->set_id(1);
    view1->set_name("001");


    mve::Scene::ViewList & views = scene->get_views();
    views.push_back(view0);
    views.push_back(view1);

    float rot0[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    float trans0[9] = {0.0, 0.0, 0.0};

    float rot1[9] = {0.9958143234, -0.09047859907, -0.02066593803, 0.0904353857,
    0.996034503, -0.003206958761, 0.02082847804, 0.001360671129, 0.9998072386};
//    float rot1[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    float trans1[9] = {0.3, 0.0, 0.0};

    mve::CameraInfo cam0;
    cam0.flen = 1.0;
    std::copy(rot0, rot0 + 9, cam0.rot);
    std::copy(trans0, trans0 + 3, cam0.trans);

    mve::CameraInfo cam1;
    cam1.flen = 1.0;
    std::copy(rot1, rot1 + 9, cam1.rot);
    std::copy(trans1, trans1 + 3, cam1.trans);

    view0->set_camera(cam0);
    view1->set_camera(cam1);

    int dim = 460;
    int gridsize = 15;

    mve::ByteImage::Ptr image0 = mve::ByteImage::create(dim, dim, 1);
    mve::ByteImage::Ptr image1 = mve::ByteImage::create(dim, dim, 1);
    image0->fill(80);
    image1->fill(100);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            if (std::abs(i/gridsize - j/gridsize) % 2 == 0)
                   image0->at(i, j, 0) = 120;
    view0->set_image(image0, "undistorted");

    mve::FloatImage::Ptr depth0 = mve::FloatImage::create(dim, dim, 1);
    mve::FloatImage::Ptr depth1 = mve::FloatImage::create(dim, dim, 1);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
                depth1->at(i, j, 0) = 5.0 + 0.005 * i + 0.005 * j;

    math::Matrix3f mat;
    math::Vec3f vec;
    cam1.fill_reprojection(cam0, dim, dim, dim, dim, *mat, *vec);

    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
        {
            math::Vec3f orig((float)i + 0.5, (float)j + 0.5, 1.0);
            math::Vec3f proj = mat * orig * depth1->at(i, j, 0) + vec;
            proj[0] /= proj[2];
            proj[1] /= proj[2];
            proj[0] -= 0.5;
            proj[1] -= 0.5;
            if (proj[0] > 0.0 && proj[0] < dim
                && proj[1] > 0.0 && proj[1] < dim)
            {
                image1->at(i, j, 0) = image0->linear_at(proj[0], proj[1], 0);
                depth0->at(proj[0], proj[1], 0) = proj[2];
            }

        }
    view1->set_image(image1, "undistorted");


    math::Matrix3f inv;

    cam0.fill_inverse_calibration(*inv, dim, dim);
    mve::image::depthmap_convert_conventions<float>(depth0, inv, true);
    view0->set_image(depth0, "depth-L0");

    cam1.fill_inverse_calibration(*inv, dim, dim);
    mve::image::depthmap_convert_conventions<float>(depth1, inv, true);
    view1->set_image(depth1, "depth-L0");

    view0->save_view_as(util::fs::join_path(base_views, "view_0000.mve"));
    view1->save_view_as(util::fs::join_path(base_views, "view_0001.mve"));

    int scale = 5;

    /* run optimization */
    StereoView::Ptr main_view = StereoView::create(view0, "undistorted");
    main_view->set_scale(scale);
    std::vector<StereoView::Ptr> neighbors;
    neighbors.emplace_back(StereoView::create(view1, "undistorted"));
    neighbors.back()->set_scale(scale);

    Surface::Ptr surface = Surface::create_planar(6.0, dim, dim, scale);
    DepthOptimizer::Options do_opts;
    do_opts.regularization = 0.001;
    do_opts.min_scale = 4;
    do_opts.num_iterations = 10;
    do_opts.debug_lvl = 3;
    DepthOptimizer optimizer(main_view, neighbors, surface, do_opts);

    view0->set_image(optimizer.get_depth(), "smvs-initial");
    util::WallTimer timer;
    optimizer.optimize();
    std::cout << "Optimization took: " << timer.get_elapsed_sec() << "sec"
        << std::endl;
    view0->set_image(optimizer.get_depth(), "smvs-opt");

    scene->save_scene();
}

