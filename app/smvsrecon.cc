/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

#include "mve/scene.h"
#include "mve/view.h"
#include "mve/image_tools.h"
#include "mve/mesh_io.h"
#include "mve/mesh_io_ply.h"
#include "util/arguments.h"
#include "util/file_system.h"
#include "util/string.h"
#include "util/tokenizer.h"
#include "util/system.h"
#include "util/timer.h"
#include "math/octree_tools.h"

#include "thread_pool.h"
#include "stereo_view.h"
#include "depth_optimizer.h"
#include "mesh_generator.h"
#include "view_selection.h"
#include "sgm_stereo.h"

/* -------------------------------------------------------------------------- */

struct AppSettings
{
    std::string scene_dname;
    std::string image_embedding;
    std::vector<int> view_ids;
    float regularization;
    int output_scale;
    int input_scale;
    int debug_lvl;
    std::size_t num_neighbors;
    std::size_t min_neighbors;
    std::size_t num_threads;
    bool use_shading;
    float light_surf_regularization;
    bool gamma_correction;
    bool recon_only;
    bool cut_surface;
    float simplify;
    bool create_triangle_mesh;
    bool use_sgm;
    float sgm_min;
    float sgm_max;
    std::string sgm_range;
    bool force_recon;
    bool force_sgm;
    bool clean_scene;

    AppSettings()
        : image_embedding("undistorted")
        , regularization(1.0f)
        , output_scale(2)
        , input_scale(-1)
        , debug_lvl(0)
        , num_neighbors(6)
        , min_neighbors(3)
        , light_surf_regularization(0)
        , simplify(0.0f)
        , sgm_min(0.0f)
        , sgm_max(0.0f)
        , sgm_range("")
    {
        num_threads = std::thread::hardware_concurrency();
        recon_only = false;
        use_shading = false;
        use_sgm = true;
        force_recon = false;
        force_sgm = false;
        cut_surface = true;
        gamma_correction = false;
        clean_scene = false;
        create_triangle_mesh = false;
    }
};

AppSettings
args_to_settings(int argc, char** argv)
{
    /* Setup argument parser. */
    util::Arguments args;
    args.set_exit_on_error(true);
    args.set_nonopt_minnum(1);
    args.set_nonopt_maxnum(1);
    args.set_helptext_indent(25);
    args.set_usage("Usage: smvsrecon [ OPTS ] SCENE_DIR ");
    args.set_description("Shading aware Multi-View Stereo");

    args.add_option('a', "alpha", true, "Regularization parameter, a higher "
        "alpha leads to smoother surfaces [1]");
    args.add_option('s', "scale", true, "Scale of input images [estimated to "
        "reduce images to a maximum of ~1.5MP]");
    args.add_option('i', "image", true, "Image embedding [undistorted]");
    args.add_option('n', "neighbors", true, "Number of neighbors for "
        "recon [6]");
    args.add_option('o', "output-scale", true, "Scale of output depth [2]");
    args.add_option('l', "list-view", true,   "Reconstructs given view"
        "IDs (given as string \"0-10\")");
    args.add_option('t', "threads", true, "Number of threads [Num CPU cores]. "
        "Peak memory requirement is ~1GB per thread and 2 megapixel image "
        "resolution");
    args.add_option('d', "debug-lvl", true, "Debug level [0]");
    args.add_option('r', "recon-only", false, "Generate only depth maps "
        "and no output ply. [off]");
    args.add_option('S', "shading", false, "Use shading-based optimization. "
        "[off]");
    args.add_option('R', "regularize-lighting", true, "Use additional basic "
        "surface regularization when lighting is turned on. This is untested "
        "but may improve results for scenes with complex lighting "
        "(0 = off, 100 = use full basic regularizer). [0]");
    args.add_option('g', "gamma-srgb", false, "Apply inverse SRGB gamma"
        " correction. [off]");
    args.add_option('m', "mesh", false, "Create Triangle mesh "
        "instead of simple point cloud (WIP). [off]");
    args.add_option('\0', "no-cut", false, "Turn off surface cutting and"
        " export fill pointcloud from all depth values. [on]");
    args.add_option('\0', "simplify", true, "Simplify triangle mesh "
        "(WIP). Given as percentage [100 = keep everything]");
    args.add_option('\0', "min-neighbors", true, "Minimal number of "
        "neighbors for reconstruction. [3]");
    args.add_option('\0', "force", false, "Force reconstruction of "
        "result embeddings");
    args.add_option('\0', "no-sgm", false, "Turn off semi-global "
        "matching.");
    args.add_option('\0', "force-sgm", false, "Force reconstruction of "
        "SGM embeddings.");
    args.add_option('\0', "sgm-range", true, "Range for SGM depth sweep, "
        "given as string \"0.1,3.5\" "
        "[estimated from SfM pointcloud] "
        "(this option is untested please report any issues)");
    args.add_option('\0', "clean", false, "Clean scene and remove all "
        "result embeddings");

    args.parse(argc, argv);

    /* Init default settings. */
    AppSettings conf;
    conf.scene_dname = args.get_nth_nonopt(0);

    /* Scan arguments. */
    while (util::ArgResult const* arg = args.next_result())
    {
        if (arg->opt == NULL)
            continue;

        if (arg->opt->lopt == "alpha")
            conf.regularization= arg->get_arg<float>();
        else if (arg->opt->lopt == "scale")
            conf.input_scale = arg->get_arg<int>();
        else if (arg->opt->lopt == "image")
            conf.image_embedding = arg->arg;
        else if (arg->opt->lopt == "neighbors")
            conf.num_neighbors = arg->get_arg<int>();
        else if (arg->opt->lopt == "output-scale")
            conf.output_scale = arg->get_arg<int>();
        else if (arg->opt->lopt == "list-view")
            args.get_ids_from_string(arg->arg, &conf.view_ids);
        else if (arg->opt->lopt == "threads")
            conf.num_threads = arg->get_arg<std::size_t>();
        else if (arg->opt->lopt == "debug-lvl")
            conf.debug_lvl = arg->get_arg<unsigned int>();
        else if (arg->opt->lopt == "min-neighbors")
            conf.min_neighbors = arg->get_arg<std::size_t>();
        else if (arg->opt->lopt == "simplify")
            conf.simplify = arg->get_arg<float>();
        else if (arg->opt->lopt == "shading")
            conf.use_shading = true;
        else if (arg->opt->lopt == "regularize-lighting")
            conf.light_surf_regularization = arg->get_arg<float>();
        else if (arg->opt->lopt == "gamma-srgb")
            conf.gamma_correction = true;
        else if (arg->opt->lopt == "recon-only")
            conf.recon_only = true;
        else if (arg->opt->lopt == "mesh")
            conf.create_triangle_mesh = true;
        else if (arg->opt->lopt == "force")
            conf.force_recon = true;
        else if (arg->opt->lopt == "no-cut")
            conf.cut_surface = false;
        else if (arg->opt->lopt == "no-sgm")
            conf.use_sgm = false;
        else if (arg->opt->lopt == "force-sgm")
            conf.force_sgm = true;
        else if (arg->opt->lopt == "sgm-range")
            conf.sgm_range = arg->arg;
        else if (arg->opt->lopt == "clean")
            conf.clean_scene = true;
        else
            throw std::runtime_error("Unknown option");
    }

    /* Process and cleanup arguments */
    if (conf.min_neighbors < 1)
    {
        std::cout << "[Warning] Need at least 1 neighbor for reconstruction, "
            << "setting min-neighbors to 1." << std::endl;
        conf.min_neighbors = 1;
    }
    if (conf.output_scale < 1)
    {
        std::cout << "[Warning] Output scale cannot be smaller than 1, "
            << "setting output-scale to 1." << std::endl;
        conf.output_scale = 1;
    }

    if (!conf.create_triangle_mesh && conf.simplify > 0.0f)
        std::cout << "[Warning] Only mesh output can be simplified. "
            << "Ignoring simplify argument!" << std::endl;

    if (conf.create_triangle_mesh && !conf.cut_surface)
        std::cout << "[Warning] Turning surface cutting off for mesh output"
            << " is not the best idea - the mesh will be huge!" << std::endl;

    if (conf.sgm_range.size() > 0)
    {
        util::Tokenizer tok;
        tok.split(conf.sgm_range, ',');
        if (tok.size() != 2)
        {
            std::cerr << "[Error] Invalid SGM Range. Exiting." << std::endl;
            std::exit(EXIT_FAILURE);
        }
        conf.sgm_min = tok.get_as<float>(0);
        conf.sgm_max = tok.get_as<float>(1);
    }

    if (conf.force_sgm && !conf.use_sgm)
    {
        std::cerr << "[Warning] Force-SGM is activated but SGM is deactivated."
            " Activating SGM automatically." << std::endl;
        conf.use_sgm = true;
    }
    if (!conf.use_sgm)
        std::cout << "Initializing depth without SGM." << std::endl;

    if (conf.force_sgm)
        conf.force_recon = true;

    return conf;
}

/* -------------------------------------------------------------------------- */

void generate_mesh (AppSettings const& conf, mve::Scene::Ptr scene,
    std::string const& input_name, std::string const& dm_name)
{
    std::cout << "Generating ";
    if (conf.create_triangle_mesh)
        std::cout << "Mesh";
    else
        std::cout << "Pointcloud";
    if (conf.cut_surface)
        std::cout << ", Cutting surfaces";

    util::WallTimer timer;
    mve::Scene::ViewList recon_views;
    for (int i : conf.view_ids)
        recon_views.push_back(scene->get_views()[i]);

    std::cout << " for " << recon_views.size() << " views ..." << std::endl;

    smvs::MeshGenerator::Options meshgen_opts;
    meshgen_opts.num_threads = conf.num_threads;
    meshgen_opts.cut_surfaces = conf.cut_surface;
    meshgen_opts.simplify = conf.simplify;
    meshgen_opts.create_triangle_mesh = conf.create_triangle_mesh;

    smvs::MeshGenerator meshgen(meshgen_opts);
    mve::TriangleMesh::Ptr mesh = meshgen.generate_mesh(recon_views,
        input_name, dm_name);

    std::cout << "Done. Took: " << timer.get_elapsed_sec() << "s" << std::endl;

    if (conf.create_triangle_mesh)
        mesh->recalc_normals();

    std::string meshname;
    if (conf.use_shading)
        meshname =
            util::fs::join_path(scene->get_path(), "smvs-S.ply");
    else
        meshname =
            util::fs::join_path(scene->get_path(), "smvs-B.ply");
    mve::geom::SavePLYOptions opts;
    opts.write_vertex_normals = true;
    opts.write_vertex_values = true;
    opts.write_vertex_confidences = true;
    mve::geom::save_ply_mesh(mesh, meshname, opts);
}

/* -------------------------------------------------------------------------- */

void reconstruct_sgm_depth_for_view (AppSettings const& conf,
    smvs::StereoView::Ptr main_view,
    std::vector<smvs::StereoView::Ptr> neighbors,
    mve::Bundle::ConstPtr bundle = nullptr)
{
    smvs::SGMStereo::Options sgm_opts;
    sgm_opts.scale = 1;
    sgm_opts.num_steps = 128;
    sgm_opts.debug_lvl = conf.debug_lvl;
    sgm_opts.min_depth = conf.sgm_min;
    sgm_opts.max_depth = conf.sgm_max;

    util::WallTimer sgm_timer;
    mve::FloatImage::Ptr d1 = smvs::SGMStereo::reconstruct(sgm_opts, main_view,
        neighbors[0], bundle);
    if (neighbors.size() > 1)
    {
        mve::FloatImage::Ptr d2 = smvs::SGMStereo::reconstruct(sgm_opts,
            main_view, neighbors[1], bundle);
        for (int p = 0; p < d1->get_pixel_amount(); ++p)
        {
            if (d2->at(p) == 0.0f)
                continue;
            if (d1->at(p) == 0.0f)
            {
                d1->at(p) = d2->at(p);
                continue;
            }

            float max = std::max(d1->at(p), d2->at(p));
            float min = std::min(d1->at(p), d2->at(p));
            if (min / max < 0.95)
                d1->at(p) = 0.0f;
            else
                d1->at(p) = (d1->at(p) + d2->at(p)) * 0.5;
        }
    }
    mve::FloatImage::Ptr init = mve::FloatImage::create(
        main_view->get_width(), main_view->get_height(), 1);
    mve::image::rescale_nearest<float>(d1, init);

    if (conf.debug_lvl > 0)
        std::cout << "SGM took: " << sgm_timer.get_elapsed_sec()
        << "sec" << std::endl;

    main_view->write_depth_to_view(init, "sgm-depth");
}

/* -------------------------------------------------------------------------- */

int
main (int argc, char** argv)
{
    util::system::register_segfault_handler();
    util::system::print_build_timestamp("Shading-aware Multi-view Stereo");

    AppSettings conf = args_to_settings(argc, argv);
    std::cout << std::endl;

    /* Start processing */

    /* Load scene */
    mve::Scene::Ptr scene(mve::Scene::create());
    scene->load_scene(conf.scene_dname);
    mve::Scene::ViewList& views(scene->get_views());

    /* Check bundle file */
    mve::Bundle::ConstPtr bundle;
    try
    {
        bundle = scene->get_bundle();
    }
    catch (std::exception e)
    {
        bundle = nullptr;
        std::cout << "Cannot load bundle file, forcing SGM." << std::endl;
        conf.use_sgm = true;
        if (conf.sgm_max == 0.0)
        {
            std::cout << "Error: No bundle file and SGM depth given, "
            "please use the --sgm-range option to specify the "
            "depth sweep range for SGM."<< std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    /* Reconstruct all views if no specific list is given */
    if (conf.view_ids.empty())
        for (auto view : views)
            if (view != nullptr && view->is_camera_valid())
                conf.view_ids.push_back(view->get_id());

    if (conf.clean_scene)
    {
        std::cout << "Cleaning Scene, removing all result embeddings."
            << std::endl;
        for (std::size_t i = 0; i < views.size(); ++i)
        {
            mve::View::Ptr view = views[i];
            if (view == nullptr)
                continue;
            mve::View::ImageProxies proxies = view->get_images();
            for (std::size_t p = 0; p < proxies.size(); ++p)
            {
                std::string name = proxies[p].name;
                std::string left = util::string::left(name, 4);
                if (left.compare("smvs") == 0
                    || name.compare("sgm-depth") == 0
                    || name.compare("lighting-shaded") == 0
                    || name.compare("lighting-sphere") == 0
                    || name.compare("implicit-albedo") == 0)
                    view->remove_image(name);
            }
            view->save_view();
        }
        std::exit(EXIT_SUCCESS);
    }

    /* Scale input images */
    if (conf.input_scale < 0)
    {
        double avg_image_size = 0;
        int view_counter = 0;
        for (std::size_t i = 0; i < views.size(); ++i)
        {
            mve::View::Ptr view = views[i];
            if (view == nullptr || !view->has_image(conf.image_embedding))
                continue;

            mve::View::ImageProxy const* proxy =
                view->get_image_proxy(conf.image_embedding);
            uint32_t size = proxy->width * proxy->height;
            avg_image_size += static_cast<double>(size);
            view_counter += 1;
        }
        avg_image_size /= static_cast<double>(view_counter);
        
        int const max_image_size = 1.7e6;
        if (avg_image_size > max_image_size)
            conf.input_scale = std::ceil(std::log2(
                avg_image_size / max_image_size) / 2);
        else
            conf.input_scale = 0;
        std::cout << "Automatic input scale: " << conf.input_scale << std::endl;
    }
    std::string input_name;
    if (conf.input_scale > 0)
        input_name = "undist-L" + util::string::get(conf.input_scale);
    else
        input_name = conf.image_embedding;
    std::cout << "Input embedding: " << input_name << std::endl;

    std::string output_name;
    if (conf.use_shading)
        output_name = "smvs-S" + util::string::get(conf.input_scale);
    else
        output_name = "smvs-B" + util::string::get(conf.input_scale);
    std::cout << "Output embedding: " << output_name << std::endl;

    /* Add views to reconstruction list */
    std::vector<int> reconstruction_list;
    int already_reconstructed = 0;
    for (std::size_t v = 0; v < conf.view_ids.size(); ++v)
    {
        int const i = conf.view_ids[v];
        if (views[i] == nullptr)
        {
            std::cout << "View ID " << i << " invalid, skipping view."
            << std::endl;
            continue;
        }
        if (!views[i]->has_image(conf.image_embedding))
        {
            std::cout << "View ID " << i << " missing image embedding, "
            << "skipping view." << std::endl;
            continue;
        }

        /* Only reconstruct missing views or if forced */
        if (conf.force_recon || !views[i]->has_image(output_name))
            reconstruction_list.push_back(i);
        else
            already_reconstructed += 1;
    }
    if (already_reconstructed > 0)
        std::cout << "Skipping " << already_reconstructed
        << " views that are already reconstructed." << std::endl;

    /* Create reconstruction threads */
    ThreadPool thread_pool(std::max<std::size_t>(std::min(conf.num_threads,
        reconstruction_list.size()), 1));

    /* Create input embedding */
    std::vector<std::future<void>> resize;
    for (std::size_t i = 0; i < views.size(); ++i)
    resize.emplace_back(thread_pool.add_task(
        [i, &views, &input_name, &conf]
    {
        mve::View::Ptr view = views[i];
        if (view == nullptr
            || !view->has_image(conf.image_embedding)
            || view->has_image(input_name))
            return;

        mve::ByteImage::ConstPtr input =
            view->get_byte_image(conf.image_embedding);
        mve::ByteImage::Ptr scaled = input->duplicate();
        for (int i = 0; i < conf.input_scale; ++i)
            scaled = mve::image::rescale_half_size_gaussian<uint8_t>(scaled);
        view->set_image(scaled, input_name);
        view->save_view();
    }));
    for(auto && resized: resize) resized.get();

    std::vector<std::future<void>> results;
    std::mutex counter_mutex;
    std::size_t started = 0;
    std::size_t finished = 0;
    util::WallTimer timer;
    smvs::ViewSelection::Options view_select_opts;
    view_select_opts.num_neighbors = conf.num_neighbors;
    view_select_opts.embedding = input_name;
    smvs::ViewSelection view_selection(view_select_opts, views, bundle);

    for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
    {
        int const i = reconstruction_list[v];

        results.emplace_back(thread_pool.add_task(
            [v, i, &views, &conf, &counter_mutex, &input_name, &output_name,
             &started, &finished, &reconstruction_list, &view_selection,
             bundle, scene]
        {
            smvs::StereoView::Ptr main_view = smvs::StereoView::create(views[i],
                input_name, conf.gamma_correction);

            mve::Scene::ViewList neighbors =
                view_selection.get_neighbors_for_view(i);

            if (neighbors.size() < conf.min_neighbors)
            {
                std::unique_lock<std::mutex> lock2(counter_mutex);
                std::cout << "View ID: " << i << " not enough neighbors, "
                    "skipping view." << std::endl;
                finished += 1;
                lock2.unlock();
                return;
            }

            std::vector<smvs::StereoView::Ptr> stereo_views;

            std::unique_lock<std::mutex> lock(counter_mutex);
            std::cout << "\rStarting "
                << ++started << "/" << reconstruction_list.size()
                << " ID: " << i
                << " Neighbors: ";
            for (std::size_t n = 0; n < conf.num_neighbors
                && n < neighbors.size() ; ++n)
                std::cout << neighbors[n]->get_id() << " ";
            std::cout << std::endl;
            lock.unlock();

            for (std::size_t n = 0; n < conf.num_neighbors
                && n < neighbors.size() ; ++n)
            {
                smvs::StereoView::Ptr sv = smvs::StereoView::create(
                    neighbors[n], input_name, conf.gamma_correction);
                stereo_views.push_back(sv);
            }

            if (conf.use_sgm)
                if (conf.force_sgm || !views[i]->has_image("sgm-depth")
                    || views[i]->get_image_proxy("sgm-depth")->width !=
                    views[i]->get_image_proxy(input_name)->width
                    || views[i]->get_image_proxy("sgm-depth")->height !=
                    views[i]->get_image_proxy(input_name)->height)
                    reconstruct_sgm_depth_for_view(conf, main_view,
                        stereo_views, bundle);

            smvs::DepthOptimizer::Options do_opts;
            do_opts.regularization = 0.01 * conf.regularization;
            do_opts.num_iterations = 5;
            do_opts.debug_lvl = conf.debug_lvl;
            do_opts.min_scale = conf.output_scale;
            do_opts.use_shading = conf.use_shading;
            do_opts.output_name = output_name;
            do_opts.use_sgm = conf.use_sgm;
            do_opts.light_surf_regularization = conf.light_surf_regularization;

            smvs::DepthOptimizer optimizer(main_view, stereo_views,
                bundle, do_opts);
            optimizer.optimize();

            std::unique_lock<std::mutex> lock2(counter_mutex);
            std::cout << "\rFinished "
                << ++finished << "/" << reconstruction_list.size()
                << " ID: " << i
                << std::endl;
            lock2.unlock();
        }));
    }
    /* Wait for reconstruction to finish */
    for(auto && result: results) result.get();

    /* Save results */
    if (results.size() > 0)
    {
        scene->save_views();
        std::cout << "Done. Reconstruction took: "
            << timer.get_elapsed_sec() << "s" << std::endl;
    }
    else
        std::cout << "All valid views are already reconstructed." << std::endl;

    /* Generate mesh */
    if (!conf.recon_only)
        generate_mesh(conf, scene, input_name, output_name);

    std::exit(EXIT_SUCCESS);
}
