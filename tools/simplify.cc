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

#include "mve/mesh_io.h"
#include "mve/mesh_io_ply.h"
#include "util/arguments.h"
#include "util/file_system.h"
#include "util/strings.h"
#include "util/tokenizer.h"
#include "util/system.h"
#include "util/timer.h"

#include "thread_pool.h"
#include "mesh_simplifier.h"

/* -------------------------------------------------------------------------- */

struct AppSettings
{
    std::string mesh_name;
    std::string out_name;
    std::size_t num_threads;
    float percent = 70.0f;

    AppSettings (void)
    {
        num_threads = std::thread::hardware_concurrency();
    }
};

AppSettings
args_to_settings(int argc, char** argv)
{
    /* Setup argument parser. */
    util::Arguments args;
    args.set_exit_on_error(true);
    args.set_nonopt_minnum(2);
    args.set_nonopt_maxnum(2);
    args.set_helptext_indent(25);
    args.set_usage("Usage: simplify [ OPTS ] MESH OUT_MESH ");
    args.set_description("Simplify Mesh");

    args.add_option('p', "percent", true, "Simplify to certain percent"
        "(WIP) [70 = keep 70%]");

    args.parse(argc, argv);

    /* Init default settings. */
    AppSettings conf;
    conf.mesh_name = args.get_nth_nonopt(0);
    conf.out_name = args.get_nth_nonopt(1);

    /* Scan arguments. */
    while (util::ArgResult const* arg = args.next_result())
    {
        if (arg->opt == NULL)
            continue;

        if (arg->opt->lopt == "percent")
            conf.percent= arg->get_arg<float>();
        else
            throw std::runtime_error("Unknown option");
    }

    /* Process and cleanup arguments */
    if (conf.percent < 0 || conf.percent > 100)
    {
        std::cerr << "[Error] Invalid percent number. Exiting." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return conf;
}

/* -------------------------------------------------------------------------- */

int
main (int argc, char** argv)
{
    util::system::register_segfault_handler();
    util::system::print_build_timestamp("Simplify Mesh");

    AppSettings conf = args_to_settings(argc, argv);
    std::cout << std::endl;

    /* Load meshes */
    mve::TriangleMesh::Ptr mesh = mve::geom::load_mesh(conf.mesh_name);
    
    /* Start processing */
    std::cout << "Processing ... ";

    smvs::MeshSimplifier simplifier(mesh);
    mesh = simplifier.get_simplified(conf.percent);
    
    std::cout << "Done." << std::endl;

    /* Save output */
    mve::geom::SavePLYOptions opts;
    mve::geom::save_ply_mesh(mesh, conf.out_name, opts);

    std::exit(EXIT_SUCCESS);
}
