/*
 * Copyright (c) 2016-2017, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <iostream>
#include <stdexcept>
#include <vector>
#include <set>
#include <string>

#include "mve/scene.h"
#include "mve/view.h"
#include "mve/image_tools.h"
#include "mve/mesh_io.h"
#include "mve/mesh_io_ply.h"
#include "util/arguments.h"
#include "util/file_system.h"
#include "util/strings.h"
#include "util/tokenizer.h"
#include "util/system.h"
#include "util/timer.h"

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
    std::vector<int> view_ids;
    std::string image_embedding = "undistorted";
    float regularization = 1.0;
    int output_scale = 2;
    int input_scale = -1;
    int debug_lvl = 0;
    std::size_t num_neighbors = 6;
    std::size_t min_neighbors = 3;
    std::size_t max_pixels = 1700000;
    std::size_t num_threads = std::thread::hardware_concurrency();
    bool use_shading = false;
    float light_surf_regularization = 0.0f;
    bool gamma_correction = false;
    bool recon_only = false;
    bool cut_surface = true;
    bool create_triangle_mesh = false;
    std::string aabb_string = "";
    bool simplify = false;
    bool use_sgm = true;
    float sgm_min = 0.0f;
    float sgm_max = 0.0f;
    int sgm_scale = 1;
    std::string sgm_range = "";
    bool force_recon = false;
    bool force_sgm = false;
    bool full_optimization = false;
    bool clean_scene = false;
    math::Vec3f aabb_min = math::Vec3f(0.0f);
    math::Vec3f aabb_max = math::Vec3f(0.0f);

    AppSettings (void) {}
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
    args.add_option('M', "max-pixels", true, "Maximal number of "
        "pixels for reconstruction. Images will be rescaled "
        "to be below this value. [1700000]");
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
    args.add_option('y', "simplify", false, "Create simplified triangle mesh "
        "(WIP). [off]");
    args.add_option('f', "force", false, "Force reconstruction of "
        "result embeddings");
    args.add_option('\0', "no-cut", false, "Turn off surface cutting and"
        " export fill pointcloud from all depth values. [on]");
    args.add_option('\0', "aabb", true, "Comma separated AABB for output: "
        "min,min,min,max,max,max");
    args.add_option('\0', "min-neighbors", true, "Minimal number of "
        "neighbors for reconstruction. [3]");
    args.add_option('\0', "no-sgm", false, "Turn off semi-global "
        "matching.");
    args.add_option('\0', "force-sgm", false, "Force reconstruction of "
        "SGM embeddings.");
    args.add_option('\0', "sgm-scale", true, "Scale of reconstruction of "
        "SGM embeddings relative to input scale. [1]");
    args.add_option('\0', "sgm-range", true, "Range for SGM depth sweep, "
        "given as string \"0.1,3.5\". "
        "[estimated from SfM pointcloud] "
        "(this option is untested please report any issues)");
    args.add_option('\0', "full-opt", false, "Run full optimization "
        "on all nodes (otherwise it only runs on non converged nodes) [off]");
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
        else if (arg->opt->lopt == "max-pixels")
            conf.max_pixels = arg->get_arg<std::size_t>();
        else if (arg->opt->lopt == "simplify")
            conf.simplify = true;
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
        else if (arg->opt->lopt == "aabb")
            conf.aabb_string = arg->arg;
        else if (arg->opt->lopt == "force")
            conf.force_recon = true;
        else if (arg->opt->lopt == "no-cut")
            conf.cut_surface = false;
        else if (arg->opt->lopt == "no-sgm")
            conf.use_sgm = false;
        else if (arg->opt->lopt == "force-sgm")
            conf.force_sgm = true;
        else if (arg->opt->lopt == "sgm-scale")
            conf.sgm_scale = arg->get_arg<int>();
        else if (arg->opt->lopt == "sgm-range")
            conf.sgm_range = arg->arg;
        else if (arg->opt->lopt == "full-opt")
            conf.full_optimization = true;
        else if (arg->opt->lopt == "clean")
            conf.clean_scene = true;
        else
            throw std::runtime_error("Unknown option");
    }

    /* Process and cleanup arguments */
    if (conf.num_neighbors < 1)
    {
        std::cout << "[Warning] Need at least 1 neighbor for reconstruction, "
            << "setting num-neighbors to 1." << std::endl;
        conf.num_neighbors = 1;
    }
    conf.min_neighbors = std::min(conf.min_neighbors, conf.num_neighbors);

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

    if (conf.create_triangle_mesh && !conf.cut_surface && !conf.simplify)
        std::cout << "[Warning] Turning surface cutting off for unsimplified"
            << " mesh output might create a huge file." << std::endl;

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

    if (conf.aabb_string.size() > 0)
    {
        util::Tokenizer tok;
        tok.split(conf.aabb_string, ',');
        if (tok.size() != 6)
        {
            std::cerr << "Error: AABB invalid" << std::endl;
            std::exit(EXIT_FAILURE);
        }
        for (int i = 0; i < 3; ++i)
        {
            conf.aabb_min[i] = tok.get_as<float>(i);
            conf.aabb_max[i] = tok.get_as<float>(i + 3);
        }
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

    if (conf.aabb_string.size() > 0)
    {
        std::cout << "Clipping to AABB: (" << conf.aabb_min << ") / ("
            << conf.aabb_max << ")" << std::endl;

        mve::TriangleMesh::VertexList const& verts = mesh->get_vertices();
        std::vector<bool> aabb_clip(verts.size(), false);
        for (std::size_t v = 0; v < verts.size(); ++v)
            for (int i = 0; i < 3; ++i)
                if (verts[v][i] < conf.aabb_min[i]
                    || verts[v][i] > conf.aabb_max[i])
                    aabb_clip[v] = true;
        mesh->delete_vertices_fix_faces(aabb_clip);
    }

    std::cout << "Done. Took: " << timer.get_elapsed_sec() << "s" << std::endl;

    if (conf.create_triangle_mesh)
        mesh->recalc_normals();

    /* Build mesh name */
    std::string meshname = "smvs-";
    if (conf.create_triangle_mesh)
        meshname += "m-";
    if (conf.use_shading)
        meshname += "S";
    else
        meshname += "B";
    meshname += util::string::get(conf.input_scale) + ".ply";
    meshname = util::fs::join_path(scene->get_path(), meshname);

    /* Save mesh */
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
    sgm_opts.scale = conf.sgm_scale;
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
            d1->at(p) = (d1->at(p) + d2->at(p)) * 0.5f;
        }
    }

    if (conf.debug_lvl > 0)
        std::cout << "SGM took: " << sgm_timer.get_elapsed_sec()
        << "sec" << std::endl;

    main_view->write_depth_to_view(d1, "smvs-sgm");
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
    mve::Scene::Ptr scene = mve::Scene::create(conf.scene_dname);
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

    /* Update legacy data */
    for (std::size_t i = 0; i < views.size(); ++i)
    {
        mve::View::Ptr view = views[i];
        if (view == nullptr)
            continue;
        mve::View::ImageProxies proxies = view->get_images();
        for (std::size_t p = 0; p < proxies.size(); ++p)
            if (proxies[p].name.compare("lighting-shaded") == 0
                || proxies[p].name.compare("lighting-sphere") == 0
                || proxies[p].name.compare("implicit-albedo") == 0)
                view->remove_image(proxies[p].name);
        view->save_view();
        for (std::size_t p = 0; p < proxies.size(); ++p)
            if (proxies[p].name.compare("sgm-depth") == 0)
            {
                std::string file = util::fs::join_path(
                    view->get_directory(), proxies[p].filename);
                std::string new_file = util::fs::join_path(
                   view->get_directory(), "smvs-sgm.mvei");
                util::fs::rename(file.c_str(), new_file.c_str());
            }
        view->reload_view();
    }

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
                if (left.compare("smvs") == 0)
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
        
        if (avg_image_size > conf.max_pixels)
            conf.input_scale = std::ceil(std::log2(
                avg_image_size / conf.max_pixels) / 2);
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

    /* Clean view id list */
    std::vector<int> ignore_list;
    for (std::size_t v = 0; v < conf.view_ids.size(); ++v)
    {
        int const i = conf.view_ids[v];
        if (i > static_cast<int>(views.size() - 1) || views[i] == nullptr)
        {
            std::cout << "View ID " << i << " invalid, skipping view."
                << std::endl;
            ignore_list.push_back(i);
            continue;
        }
        if (!views[i]->has_image(conf.image_embedding))
        {
            std::cout << "View ID " << i << " missing image embedding, "
                << "skipping view." << std::endl;
            ignore_list.push_back(i);
            continue;
        }
    }
    for (auto const& id : ignore_list)
        conf.view_ids.erase(std::remove(conf.view_ids.begin(),
            conf.view_ids.end(), id));

    /* Add views to reconstruction list */
    std::vector<int> reconstruction_list;
    int already_reconstructed = 0;
    for (std::size_t v = 0; v < conf.view_ids.size(); ++v)
    {
        int const i = conf.view_ids[v];
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
    ThreadPool thread_pool(std::max<std::size_t>(conf.num_threads, 1));

    /* View selection */
    smvs::ViewSelection::Options view_select_opts;
    view_select_opts.num_neighbors = conf.num_neighbors;
    view_select_opts.embedding = conf.image_embedding;
    smvs::ViewSelection view_selection(view_select_opts, views, bundle);
    std::vector<mve::Scene::ViewList> view_neighbors(
        reconstruction_list.size());
    std::vector<std::future<void>> selection_tasks;
    for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
    {
        int const i = reconstruction_list[v];
        selection_tasks.emplace_back(thread_pool.add_task(
            [i, v, &views, &view_selection, &view_neighbors]
        {
            view_neighbors[v] = view_selection.get_neighbors_for_view(i);
        }));
    }
    if (selection_tasks.size() > 0)
    {
        std::cout << "Running view selection for "
            << selection_tasks.size() << " views... " << std::flush;
        util::WallTimer timer;
        for(auto && task : selection_tasks) task.get();
        std::cout << " done, took " << timer.get_elapsed_sec()
            << "s." << std::endl;
    }
    std::vector<int> skipped;
    std::vector<int> final_reconstruction_list;
    std::vector<mve::Scene::ViewList> final_view_neighbors;
    for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
        if (view_neighbors[v].size() < conf.min_neighbors)
            skipped.push_back(reconstruction_list[v]);
        else
        {
            final_reconstruction_list.push_back(reconstruction_list[v]);
            final_view_neighbors.push_back(view_neighbors[v]);
        }
    if (skipped.size() > 0)
    {
        std::cout << "Skipping " << skipped.size() << " views with "
            << "insufficient number of neighbors." << std::endl;
        std::cout << "Skipped IDs: ";
        for (std::size_t s = 0; s < skipped.size(); ++s)
        {
            std::cout << skipped[s] << " ";
            if (s > 0 && s % 12 == 0)
                std::cout << std::endl << "     ";
        }
        std::cout << std::endl;
    }
    reconstruction_list = final_reconstruction_list;
    view_neighbors = final_view_neighbors;

    /* Create input embedding and resize */
    std::set<int> check_embedding_list;
    for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
    {
        check_embedding_list.insert(reconstruction_list[v]);
        for (auto & neighbor : view_neighbors[v])
            check_embedding_list.insert(neighbor->get_id());
    }
    std::vector<std::future<void>> resize_tasks;
    for (auto const& i : check_embedding_list)
    {
        mve::View::Ptr view = views[i];
        if (view == nullptr
            || !view->has_image(conf.image_embedding)
            || view->has_image(input_name))
            continue;

        resize_tasks.emplace_back(thread_pool.add_task(
            [view, &input_name, &conf]
        {
            mve::ByteImage::ConstPtr input =
                view->get_byte_image(conf.image_embedding);
            mve::ByteImage::Ptr scld = input->duplicate();
            for (int i = 0; i < conf.input_scale; ++i)
                scld = mve::image::rescale_half_size_gaussian<uint8_t>(scld);
            view->set_image(scld, input_name);
            view->save_view();
        }));
    }
    if (resize_tasks.size() > 0)
    {
        std::cout << "Resizing input images for "
            << resize_tasks.size() << " views... " << std::flush;
        util::WallTimer timer;
        for(auto && task : resize_tasks) task.get();
        std::cout << " done, took " << timer.get_elapsed_sec()
            << "s." << std::endl;
    }

    std::vector<std::future<void>> results;
    std::mutex counter_mutex;
    std::size_t started = 0;
    std::size_t finished = 0;
    util::WallTimer timer;

    for (std::size_t v = 0; v < reconstruction_list.size(); ++v)
    {
        int const i = reconstruction_list[v];

        results.emplace_back(thread_pool.add_task(
            [v, i, &views, &conf, &counter_mutex, &input_name, &output_name,
             &started, &finished, &reconstruction_list, &view_neighbors,
             bundle, scene]
        {
            smvs::StereoView::Ptr main_view = smvs::StereoView::create(
                views[i], input_name, conf.use_shading,
                conf.gamma_correction);
            mve::Scene::ViewList neighbors = view_neighbors[v];

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
                    neighbors[n], input_name);
                stereo_views.push_back(sv);
            }

            if (conf.use_sgm)
            {
                int sgm_width = views[i]->get_image_proxy(input_name)->width;
                int sgm_height = views[i]->get_image_proxy(input_name)->height;
                for (int scale = 0; scale < conf.sgm_scale; ++scale)
                {
                    sgm_width = (sgm_width + 1) / 2;
                    sgm_height = (sgm_height + 1) / 2;
                }
                if (conf.force_sgm || !views[i]->has_image("smvs-sgm")
                    || views[i]->get_image_proxy("smvs-sgm")->width !=
                        sgm_width
                    || views[i]->get_image_proxy("smvs-sgm")->height !=
                        sgm_height)
                    reconstruct_sgm_depth_for_view(conf, main_view,
                        stereo_views, bundle);
            }

            smvs::DepthOptimizer::Options do_opts;
            do_opts.regularization = 0.01 * conf.regularization;
            do_opts.num_iterations = 5;
            do_opts.debug_lvl = conf.debug_lvl;
            do_opts.min_scale = conf.output_scale;
            do_opts.use_shading = conf.use_shading;
            do_opts.output_name = output_name;
            do_opts.use_sgm = conf.use_sgm;
            do_opts.full_optimization = conf.full_optimization;
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
