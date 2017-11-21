/*
 * Copyright (C) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include <gtest/gtest.h>

#include "sse_vector.h"
#include "block_sparse_matrix.h"
#include "ldl_decomposition.h"

TEST(LDLDecompositionTest, inverse)
{
    double A[9] {2, -1, 0, -1, 2, -1, 0 , -1, 1};
    smvs::ldl_inverse(A, 3);
    
    double const epsilon = 1e-15;
    EXPECT_NEAR(1, A[0], epsilon);
    EXPECT_NEAR(1, A[1], epsilon);
    EXPECT_NEAR(1, A[2], epsilon);
    EXPECT_NEAR(1, A[3], epsilon);
    EXPECT_NEAR(2, A[4], epsilon);
    EXPECT_NEAR(2, A[5], epsilon);
    EXPECT_NEAR(1, A[6], epsilon);
    EXPECT_NEAR(2, A[7], epsilon);
    EXPECT_NEAR(3, A[8], epsilon);
}

TEST(SSEVectorTest, dot)
{
    smvs::SSEVector a(5);
    smvs::SSEVector b(5);

    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 1.0;
    a[4] = 2.0;

    b[0] = 1.0;
    b[1] = 2.0;
    b[2] = 3.0;
    b[3] = 1.0;
    b[4] = 2.0;

    double d = a.dot(b);
    EXPECT_DOUBLE_EQ(19.0, d);
}

TEST(SSEVectorTest, add)
{
    smvs::SSEVector a(5);
    smvs::SSEVector b(5);

    a[0] = 4.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 1.0;
    a[4] = 2.0;

    b[0] = 1.0;
    b[1] = 7.0;
    b[2] = 3.0;
    b[3] = 8.0;
    b[4] = 2.0;

    smvs::SSEVector c = a.add(b);
    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(a[i] + b[i], c[i]);
}

TEST(SSEVectorTest, subtract)
{
    smvs::SSEVector a(5);
    smvs::SSEVector b(5);

    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 10.0;
    a[4] = 2.0;

    b[0] = 11.0;
    b[1] = 2.0;
    b[2] = 30.0;
    b[3] = 1.0;
    b[4] = 29.0;

    smvs::SSEVector c = a.subtract(b);
    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(a[i] - b[i], c[i]);
}

TEST(SSEVectorTest, multiply)
{
    smvs::SSEVector a(5);

    a[0] = 12.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 10.0;
    a[4] = 2.0;

    double rhs = 4.3;

    smvs::SSEVector c = a.multiply(rhs);
    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(a[i] * rhs, c[i]);
}

TEST(SSEVectorTest, multiply_add)
{
    smvs::SSEVector a(5);
    a[0] = 12.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 10.0;
    a[4] = 2.0;

    smvs::SSEVector b(5);
    b[0] = 11.0;
    b[1] = 2.0;
    b[2] = 30.0;
    b[3] = 1.0;
    b[4] = 29.0;

    double factor = 4.3;

    smvs::SSEVector c = a.multiply_add(b, factor);
    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(a[i] + b[i] * factor, c[i]);
}

TEST(SSEVectorTest, multiply_sub)
{
    smvs::SSEVector a(5);
    a[0] = 12.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 10.0;
    a[4] = 2.0;

    smvs::SSEVector b(5);
    b[0] = 11.0;
    b[1] = 2.0;
    b[2] = 30.0;
    b[3] = 1.0;
    b[4] = 29.0;

    double factor = 4.3;

    smvs::SSEVector c = a.multiply_sub(b, factor);
    for (int i = 0; i < 5; ++i)
        EXPECT_DOUBLE_EQ(a[i] - b[i] * factor, c[i]);
}

TEST(SSEVectorTest, multiply_add_large)
{
    int const dim = 1000000;
    smvs::SSEVector a(dim);
    smvs::SSEVector b(dim);
    for (int i = 0; i < dim; ++i)
    {
        a[i] = i % 3;
        b[i] = i % 7;
    }

    double factor = 0.3;

    smvs::SSEVector c = a.multiply_add(b, factor);
    for (int i = 0; i < dim; ++i)
        EXPECT_DOUBLE_EQ(a[i] + b[i] * factor, c[i]);
}

TEST(SSEVectorTest, multiply_sub_large)
{
    int const dim = 1000000;
    smvs::SSEVector a(dim);
    smvs::SSEVector b(dim);
    for (int i = 0; i < dim; ++i)
    {
        a[i] = i % 7;
        b[i] = i % 3;
    }

    double factor = 0.3;

    smvs::SSEVector c = a.multiply_sub(b, factor);
    for (int i = 0; i < dim; ++i)
        EXPECT_DOUBLE_EQ(a[i] - b[i] * factor, c[i]);
}

TEST (BlockSparseMatrixTest, Create)
{
    smvs::BlockSparseMatrix<4> bs_matrix;
    smvs::BlockSparseMatrix<4> bs_matrix2(100,100);
}

TEST (BlockSparseMatrixTest, SetFromBlocks)
{
    typedef smvs::BlockSparseMatrix<2> BSMatrix;
    BSMatrix bs_matrix(4, 4);
    double values1[4] = {1, 2, 3, 4};
    double values2[4] = {4, 3, 2, 1};
    BSMatrix::Blocks blocks;
    blocks.emplace_back(0, 0, values1);
    blocks.emplace_back(2, 2, values2);
    bs_matrix.set_from_blocks(blocks);

    EXPECT_EQ(2, bs_matrix.num_non_zero());

    BSMatrix::Blocks blocks2;
    blocks2.emplace_back(2, 2, values2);
    blocks2.emplace_back(0, 0, values1);
    bs_matrix.set_from_blocks(blocks2);

    EXPECT_EQ(2, bs_matrix.num_non_zero());
}

TEST (BlockSparseMatrixTest, SetFromTriplets)
{
    typedef smvs::BlockSparseMatrix<2> BSMatrix;

    BSMatrix::Triplets triplets;
    triplets.emplace_back(0, 0, 11);
    triplets.emplace_back(0, 1, 12);
    triplets.emplace_back(0, 2, 13);
    triplets.emplace_back(0, 3, 14);

    triplets.emplace_back(1, 1, 22);
    triplets.emplace_back(1, 2, 23);

    triplets.emplace_back(2, 2, 33);
    triplets.emplace_back(2, 3, 34);
    triplets.emplace_back(2, 4, 35);
    triplets.emplace_back(2, 5, 36);

    triplets.emplace_back(3, 3, 44);
    triplets.emplace_back(3, 4, 45);

    triplets.emplace_back(4, 5, 56);

    triplets.emplace_back(5, 5, 66);

    BSMatrix bs_matrix(6, 6);
    bs_matrix.set_from_triplets(triplets);

    EXPECT_EQ(5, bs_matrix.num_non_zero());
}

TEST (BlockSparseMatrixTest, Multiply)
{
    typedef smvs::BlockSparseMatrix<2> BSMatrix;
    BSMatrix bs_matrix(4, 4);
    double values1[4] = {1, 2, 3, 4};
    double values2[4] = {5, 3, 2, 0};
    BSMatrix::Blocks blocks;
    blocks.emplace_back(0, 0, values1);
    blocks.emplace_back(2, 2, values2);
    bs_matrix.set_from_blocks(blocks);
    smvs::SSEVector x(4, 1);
    smvs::SSEVector rhs;
    rhs = bs_matrix.multiply(x);
    EXPECT_EQ(3, rhs[0]);
    EXPECT_EQ(7, rhs[1]);
    EXPECT_EQ(8, rhs[2]);
    EXPECT_EQ(2, rhs[3]);

    BSMatrix::Blocks blocks2;
    blocks2.emplace_back(2, 2, values2);
    blocks2.emplace_back(0, 0, values1);
    bs_matrix.set_from_blocks(blocks2);
    rhs = bs_matrix.multiply(x);
    EXPECT_EQ(3, rhs[0]);
    EXPECT_EQ(7, rhs[1]);
    EXPECT_EQ(8, rhs[2]);
    EXPECT_EQ(2, rhs[3]);

    BSMatrix::Blocks blocks3;
    blocks3.emplace_back(2, 2, values2);
    blocks3.emplace_back(0, 2, values1);
    blocks3.emplace_back(0, 0, values1);
    bs_matrix.set_from_blocks(blocks3);
    rhs = bs_matrix.multiply(x);
    EXPECT_EQ(6, rhs[0]);
    EXPECT_EQ(14, rhs[1]);
    EXPECT_EQ(8, rhs[2]);
    EXPECT_EQ(2, rhs[3]);
}

TEST (BlockSparseMatrixTest, SetFromTripletsMultiply)
{
    typedef smvs::BlockSparseMatrix<2> BSMatrix;

    BSMatrix::Triplets triplets;
    triplets.emplace_back(0, 0, 1);
    triplets.emplace_back(0, 1, 2);
    triplets.emplace_back(1, 0, 3);
    triplets.emplace_back(1, 1, 4);

    triplets.emplace_back(2, 2, 5);
    triplets.emplace_back(2, 3, 3);
    triplets.emplace_back(3, 2, 2);
    triplets.emplace_back(3, 3, 0);

    BSMatrix bs_matrix(4, 4);
    bs_matrix.set_from_triplets(triplets);
    smvs::SSEVector x(4, 1);
    smvs::SSEVector rhs;
    rhs = bs_matrix.multiply(x);
    EXPECT_EQ(3, rhs[0]);
    EXPECT_EQ(7, rhs[1]);
    EXPECT_EQ(8, rhs[2]);
    EXPECT_EQ(2, rhs[3]);

    triplets.emplace_back(2, 0, 2);
    triplets.emplace_back(2, 1, 7);
    triplets.emplace_back(3, 0, 4);
    triplets.emplace_back(3, 1, 1);

    BSMatrix bs_matrix2(4, 4);
    bs_matrix2.set_from_triplets(triplets);
    rhs = bs_matrix2.multiply(x);

    smvs::SSEVector rhs2;
    rhs2 = bs_matrix2.multiply(x);

    EXPECT_EQ(rhs2[0], rhs[0]);
    EXPECT_EQ(rhs2[1], rhs[1]);
    EXPECT_EQ(rhs2[2], rhs[2]);
    EXPECT_EQ(rhs2[3], rhs[3]);
}

TEST (BlockSparseMatrixTest, BlockInvert)
{
    typedef smvs::BlockSparseMatrix<2> BSMatrix;
    BSMatrix bs_matrix(4, 4);
    double values1[4] = {2, 0, 0, 2};
    BSMatrix::Blocks blocks;
    blocks.emplace_back(0, 0, values1);
    blocks.emplace_back(2, 2, values1);
    bs_matrix.set_from_blocks(blocks);
    bs_matrix.invert_blocks_inplace();

    smvs::SSEVector x(4, 1);
    smvs::SSEVector rhs;
    rhs = bs_matrix.multiply(x);
    EXPECT_NEAR(0.5, rhs[0], 1e-10);
    EXPECT_NEAR(0.5, rhs[1], 1e-10);
    EXPECT_NEAR(0.5, rhs[2], 1e-10);
    EXPECT_NEAR(0.5, rhs[3], 1e-10);
}


