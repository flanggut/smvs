/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_BLOCK_SPARSE_MATRIX_HEADER
#define SMVS_BLOCK_SPARSE_MATRIX_HEADER

#include <stdexcept>
#include <vector>
#include <array>
#include <algorithm>

#include "sse_vector.h"
#include "conjugate_gradient.h"
#include "ldl_decomposition.h"

/**
 *  Block Sparse matrix class.
 *  Note: Blocks are stored column-major, but blocks themselfes are row-major.
 *        This seems to be the most efficient structure for the given problem.
 */
template <int N>
class BlockSparseMatrix : public ConjugateGradient::Functor
{
public:
    /** Block with row/col index, and all values. */
    struct Block
    {
        Block (void) = default;
        Block (std::size_t row, std::size_t col, double const* values);

        std::size_t row = 0;
        std::size_t col = 0;
        std::array<double, N*N> values;
    };
    /** List of blocks. */
    typedef std::vector<Block> Blocks;


    /** Triplet helper for debugging */
    struct Triplet
    {
        Triplet (void) = default;
        Triplet (std::size_t row, std::size_t col, double value);

        std::size_t row = 0;
        std::size_t col = 0;
        double value;
    };
    /** List of Triplets */
    typedef std::vector<Triplet> Triplets;

    /** Internal DenseVector definition */
    typedef SSEVector DenseVector;

public:
    BlockSparseMatrix (void);
    BlockSparseMatrix (std::size_t rows, std::size_t cols);
    void allocate (std::size_t rows, std::size_t cols);
    void reserve (std::size_t num_elements);
    void set_from_blocks (Blocks const& blocks);
    void set_from_triplets (Triplets const& triplets);

    BlockSparseMatrix transpose (void) const;
    DenseVector multiply (DenseVector const& rhs) const;

    void invert_blocks_inplace(void);
    void add_triplet_to_blocks(Triplet const& triplet,
        std::map<std::size_t, Block> * blocks);

    std::size_t num_rows (void) const;
    std::size_t num_cols (void) const;
    std::size_t num_non_zero (void) const;
    
    double * begin (void);
    double * end (void);
    
    /** Conjugate Gradient functor definitions */
    std::size_t input_size (void) const;
    std::size_t output_size (void) const;
    
private:
    void blocks_from_triplets(Triplets const& triplets, Blocks * blocks);

private:
    std::size_t rows;
    std::size_t cols;
    std::vector<std::array<double, N*N>> values;
    std::vector<std::size_t> outer;
    std::vector<std::size_t> inner;
};

/* ------------------------ Implementation ------------------------ */

template <int N>
BlockSparseMatrix<N>::Block::Block (std::size_t row,
    std::size_t col, double const* values)
    : row(row), col(col)
{
    std::copy(values, values + N * N, this->values.begin());
}

template <int N>
BlockSparseMatrix<N>::Triplet::Triplet (std::size_t row,
    std::size_t col, double value)
    : row(row), col(col), value(value)
{
}

/* --------------------------------------------------------------- */

template <int N>
BlockSparseMatrix<N>::BlockSparseMatrix (void)
    : rows(0), cols(0)
{
}


template <int N>
BlockSparseMatrix<N>::BlockSparseMatrix (std::size_t rows, std::size_t cols)
{
    this->allocate(rows, cols);
}

template <int N>
void
BlockSparseMatrix<N>::allocate (std::size_t rows, std::size_t cols)
{
    if (rows % N != 0 || cols % N != 0)
        throw std::invalid_argument(
            "Rows and Cols need to be multiples of BlockSize.");

    this->rows = rows;
    this->cols = cols;
    this->values.clear();
    this->outer.clear();
    this->inner.clear();
    this->outer.resize((cols / N) + 1, 0);
}

template <int N>
void
BlockSparseMatrix<N>::reserve (std::size_t num_elements)
{
    this->values.reserve(num_elements);
}

template <int N>
void
BlockSparseMatrix<N>::set_from_blocks (Blocks const& blocks)
{
    /* Create a temporary transposed matrix */
    BlockSparseMatrix<N> transposed(this->cols, this->rows);
    transposed.values.resize(blocks.size());
    transposed.inner.resize(blocks.size());

    /* Initialize outer indices with amount of inner values. */
    for (std::size_t i = 0; i < blocks.size(); ++i)
        transposed.outer[blocks[i].row / N]++;

    /* Convert amounts to indices with prefix sum. */
    std::size_t sum = 0;
    std::vector<std::size_t> scratch(transposed.outer.size());
    for (std::size_t i = 0; i < transposed.outer.size(); ++i)
    {
        std::size_t const temp = transposed.outer[i];
        transposed.outer[i] = sum;
        scratch[i] = sum;
        sum += temp;
    }

    /* Add blocks, inner indices are unsorted. */
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        Block const& b = blocks[i];
        std::size_t pos = scratch[b.row / N]++;
        transposed.values[pos] = b.values;
        transposed.inner[pos] = b.col;
    }

    /* Transpose matrix, implicit sorting of inner indices. */
    *this = transposed.transpose();
}

template <int N>
void
BlockSparseMatrix<N>::add_triplet_to_blocks(Triplet const& triplet,
    std::map<std::size_t, Block> * blocks)
{
    std::size_t const block_row = triplet.row / N;
    std::size_t const block_col = triplet.col / N;
    std::size_t const block_row_offset = triplet.row % N;
    std::size_t const block_col_offset = triplet.col % N;
    std::size_t block_id = block_row + block_col * this->rows;
    
    Block &block = (*blocks)[block_id];
    if(block.row == 0)
    {
        block.row = block_row * N;
        block.col = block_col * N;
    }
    block.values[block_row_offset * N + block_col_offset] =
        triplet.value;
}


template <int N>
void
BlockSparseMatrix<N>::blocks_from_triplets(Triplets const& triplets,
    Blocks * blocks)
{
    std::map<std::size_t, Block> blockmap;
    
    for (std::size_t i = 0; i < triplets.size(); ++i)
        this->add_triplet_to_blocks(triplets[i], &blockmap);
    
    blocks->clear();
    // simple guess for total number of blocks
    blocks->reserve(triplets.size() / 16);
    for (auto const& b : blockmap)
        blocks->push_back(b.second);
}

template <int N>
void
BlockSparseMatrix<N>::set_from_triplets(Triplets const& triplets)
{
    Blocks blocks;
    this->blocks_from_triplets(triplets, &blocks);
    this->set_from_blocks(blocks);
}


template <int N>
BlockSparseMatrix<N>
BlockSparseMatrix<N>::transpose (void) const
{
    BlockSparseMatrix ret(this->cols, this->rows);
    ret.values.resize(this->num_non_zero());
    ret.inner.resize(this->num_non_zero());

    /* Compute inner sizes of transposed matrix. */
    for(std::size_t i = 0; i < this->inner.size(); ++i)
        ret.outer[this->inner[i] / N] += 1;

    /* Compute outer sizes of transposed matrix with prefix sum. */
    std::size_t sum = 0;
    std::vector<std::size_t> scratch(ret.outer.size());
    for (std::size_t i = 0; i < ret.outer.size(); ++i)
    {
        std::size_t const temp = ret.outer[i];
        ret.outer[i] = sum;
        scratch[i] = sum;
        sum += temp;
    }

    /* Write inner indices and values of transposed matrix. */
    for (std::size_t i = 0; i < this->outer.size() - 1; ++i)
        for (std::size_t j = this->outer[i]; j < this->outer[i + 1]; ++j)
        {
            std::size_t pos = scratch[this->inner[j] / N]++;
            ret.inner[pos] = i * N;
            ret.values[pos] = this->values[j];
        }

    return ret;
}

template<int N>
typename BlockSparseMatrix<N>::DenseVector
BlockSparseMatrix<N>::multiply (DenseVector const& rhs) const
{
    if (rhs.size() != this->cols)
        throw std::invalid_argument("Incompatible dimensions");

    DenseVector ret(this->rows, 0.0);
    for (std::size_t i = 0; i < this->outer.size() - 1; ++i)
        for (std::size_t id = this->outer[i]; id < this->outer[i + 1]; ++id)
        {
            std::size_t ret_id = this->inner[id];
            int block_id = 0;
            for (int block_row = 0; block_row < N; ++block_row)
            {
                for (int block_col = 0; block_col < N; ++ block_col)
                    ret[ret_id] += this->values[id][block_id++]
                        * rhs[i * N + block_col];
                ret_id += 1;
            }
        }
    return ret;
}

template<int N>
void
BlockSparseMatrix<N>::invert_blocks_inplace(void)
{
    for (std::size_t i = 0; i < this->values.size(); ++i)
    {
        std::array<double, N * N> b = values[i];
        ldl_inverse(b.begin(), N);
        bool nancheck = false;
        for (int i = 0; i < N * N; ++i)
            if (std::isnan(b[i]))
                nancheck = true;
        if(nancheck)
            continue;
        values[i] = b;
    }
}

template<int N>
inline std::size_t
BlockSparseMatrix<N>::num_rows (void) const
{
    return this->rows;
}

template<int N>
inline std::size_t
BlockSparseMatrix<N>::num_cols (void) const
{
    return this->cols;
}

template<int N>
inline std::size_t
BlockSparseMatrix<N>::num_non_zero (void) const
{
    return this->values.size();
}

template<int N>
inline double *
BlockSparseMatrix<N>::begin (void)
{
    return &this->values[0][0];
}

template<int N>
inline double *
BlockSparseMatrix<N>::end (void)
{
    return &this->values[values.size() - 1][N * N];
}


template<int N>
inline std::size_t
BlockSparseMatrix<N>::input_size (void) const
{
    return this->cols;
}

template<int N>
inline std::size_t
BlockSparseMatrix<N>::output_size (void) const
{
    return this->rows;
}

#endif /* SMVS_BLOCK_SPARSE_MATRIX_HEADER */
