/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "delaunay_2d.h"


void swap_edge (Edge::Ptr e)
{
    Edge::Ptr prev = e->o_prev();
    Edge::Ptr inv_prev = e->inv()->o_prev();
    /* remove edge and inv from their vertices */
    Edge::splice(e, prev);
    Edge::splice(e->inv(), inv_prev);
    /* add edge and inv to their next vertices ccw */
    Edge::splice(e, prev->l_next());
    Edge::splice(e->inv(), inv_prev->l_next());
    /* change vertex data */
    e->set_data(prev->dest(), inv_prev->dest());
}
