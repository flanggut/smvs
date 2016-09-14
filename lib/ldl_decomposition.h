/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_LDL_DECOMPOSITION_HEADER
#define SMVS_LDL_DECOMPOSITION_HEADER

#include <iostream>

SMVS_NAMESPACE_BEGIN

/** 
 * Combines Matrix L and diagonal vector D to LDL = L*D*L^T
 */
template<typename T>
void
combine_ldl (T const* L, T const* D, int size,
    T* LDL)
{
    std::fill(LDL, LDL + size * size, T(0));

    T const* L_trans_ptr = L;
    T const* L_row_ptr = L;
    for (int r = 0; r < size; ++r, L_row_ptr += size)
    {
        double * LDL_ptr = LDL;
        for (int c1 = 0; c1 < size; ++c1, ++L_trans_ptr)
            for (int c2 = 0; c2 < size; ++c2, ++LDL_ptr)
                (*LDL_ptr) += L_row_ptr[c2] * (*L_trans_ptr) * D[r];
    }
}

/**
 * Inverts symmetric matrix A using the LDL decomposition
 * This is essentially a variant of the Cholesky decomposition 
 * which doesn't require sqrt.
 */
template<typename T>
void
ldl_inverse(T * A, int const size)
{
    T * L = new T[size * size];
    T * D = new T[size];
    std::fill(L, L + size * size, 0.0);
    std::fill(D, D + size, 0.0);

    /* Factorize A into LDL^T */
    for (int j = 0; j < size; ++j)
    {
        D[j] = A[j * size + j];
        L[j * size + j] = 1.0;
        for (int k = 0 ; k < j; ++k)
            D[j] -= (L[j * size + k] * L[j * size + k]) * D[k];

        if (D[j] == 0.0)
            return;

        for (int i = j+1; i < size; ++i)
        {
            L[i * size + j] = A[i * size + j];
            for (int k = 0 ; k < j; ++k)
                L[i * size + j] -= L[i * size + k] * D[k] * L[j * size + k];
            L[i * size + j] /= D[j];
        }
    }

    /* Invert L */
    for (int i = 0; i < size; ++i)
        for (int j = i+1; j < size; ++j)
        {
            T sum(0);
            for (int k = i ; k < j; ++k)
                sum -= L[j * size + k] * L[k * size + i];
            L[j * size + i] = sum;
        }

    /* Invert D */
    for (int i = 0; i < size; ++i)
        D[i] = 1.0 / D[i];

    /* Combine Matrices */
    combine_ldl(L, D, size, A);

    /* Cleanup memory */
    delete[] L;
    delete[] D;
}

SMVS_NAMESPACE_END

#endif /* LDL_DECOMPOSITION_HEADER */
