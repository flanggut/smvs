/*
 * Copyright (c) 2017, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#include "sse_vector.h"

#include <stdexcept>
#include <smmintrin.h> // SSE4_1

#define ENABLE_SSE_VECTOR 1

SMVS_NAMESPACE_BEGIN

double
SSEVector::dot (SSEVector const& rhs) const
{
    if (this->size() != rhs.size())
        throw std::invalid_argument("Incompatible vector dimensions");

    double ret = 0;
#if ENABLE_SSE_VECTOR && defined(__SSE4_1__)
    std::size_t const dim = this->size() / 2;
    __m128d const* ptr = reinterpret_cast<__m128d const*>(this->begin());
    __m128d const* rhs_ptr =  reinterpret_cast<__m128d const*>(rhs.begin());

    for (std::size_t i = 0; i < dim; ++i, ++ptr, ++rhs_ptr)
        ret += _mm_cvtsd_f64(_mm_dp_pd(*ptr, *rhs_ptr, 0xFF));

    for (std::size_t i = this->size() % 2; i > 0; --i)
        ret += this->values[this->size() - i] * rhs[this->size() - i];
#else
    for (std::size_t i = 0; i < this->size(); ++i)
        ret += this->values[i] * rhs.values[i];
#endif
    return ret;
}

SSEVector
SSEVector::add (SSEVector const& rhs) const
{
    if (this->size() != rhs.size())
        throw std::invalid_argument("Incompatible vector dimensions");

    SSEVector result(this->size());
#if ENABLE_SSE_VECTOR && defined(__SSE4_1__)
    __m128d const* ptr = reinterpret_cast<__m128d const*>(this->begin());
    __m128d * res_ptr = reinterpret_cast<__m128d *>(result.begin());
    __m128d const* rhs_ptr =  reinterpret_cast<__m128d const*>(rhs.begin());

    __m128d _dest1;
    __m128d _dest2;

    for (std::size_t i = 0; i < this->size() - 3;
        i += 4, rhs_ptr += 2, ptr += 2, res_ptr+= 2)
    {
        _dest1 = _mm_add_pd(*ptr,  *rhs_ptr);
        _dest2 = _mm_add_pd(*(ptr + 1), *(rhs_ptr + 1));

        *res_ptr = _dest1;
        *(res_ptr + 1) = _dest2;
    }
    for (std::size_t i = this->size() % 4; i > 0; --i)
        result[this->size() - i] = this->values[this->size() - i]
            + rhs.values[this->size() - i];
#else
    for (std::size_t i = 0; i < this->size(); ++i)
        result[i] = this->values[i] + rhs.values[i];
#endif
    return result;
}

SSEVector
SSEVector::subtract (SSEVector const& rhs) const
{
    if (this->size() != rhs.size())
        throw std::invalid_argument("Incompatible vector dimensions");

    SSEVector result(this->size());
#if ENABLE_SSE_VECTOR && defined(__SSE4_1__)
    __m128d const* ptr = reinterpret_cast<__m128d const*>(this->begin());
    __m128d * res_ptr = reinterpret_cast<__m128d *>(result.begin());
    __m128d const* rhs_ptr =  reinterpret_cast<__m128d const*>(rhs.begin());

    __m128d _dest1;
    __m128d _dest2;

    for (std::size_t i = 0; i < this->size() - 3;
         i += 4, rhs_ptr += 2, ptr += 2, res_ptr+= 2)
    {
        _dest1 = _mm_sub_pd(*ptr,  *rhs_ptr);
        _dest2 = _mm_sub_pd(*(ptr + 1), *(rhs_ptr + 1));

        *res_ptr = _dest1;
        *(res_ptr + 1) = _dest2;
    }
    for (std::size_t i = this->size() % 4; i > 0; --i)
        result[this->size() - i] = this->values[this->size() - i]
            - rhs.values[this->size() - i];
#else
    for (std::size_t i = 0; i < this->size(); ++i)
        result[i] = this->values[i] - rhs.values[i];
#endif
    return result;

}

SSEVector
SSEVector::multiply (double const rhs) const
{
    SSEVector result(this->size());
#if ENABLE_SSE_VECTOR && defined(__SSE4_1__)
    __m128d const* ptr = reinterpret_cast<__m128d const*>(this->begin());
    __m128d * res_ptr = reinterpret_cast<__m128d *>(result.begin());
    __m128d const a = _mm_set_pd(rhs, rhs);

    __m128d _dest1;
    __m128d _dest2;

    for (std::size_t i = 0; i < this->size() - 3;
         i += 4, ptr += 2, res_ptr+= 2)
    {
        _dest1 = _mm_mul_pd(a, *ptr );
        _dest2 = _mm_mul_pd(a, *(ptr + 1));

        *res_ptr = _dest1;
        *(res_ptr + 1) = _dest2;
    }
    for (std::size_t i = this->size() % 4; i > 0; --i)
        result[this->size() - i] = this->values[this->size() - i] * rhs;
#else
    for (std::size_t i = 0; i < this->size(); ++i)
        result[i] = this->values[i] * rhs;
#endif
    return result;
}

SSEVector
SSEVector::multiply_add (smvs::SSEVector const& rhs, double const rhs_d) const
{
    SSEVector result(this->size());
#if ENABLE_SSE_VECTOR && defined(__SSE4_1__)
    __m128d const* ptr = reinterpret_cast<__m128d const*>(this->begin());
    __m128d * res_ptr = reinterpret_cast<__m128d *>(result.begin());
    __m128d const* rhs_ptr =  reinterpret_cast<__m128d const*>(rhs.begin());
    __m128d const a = _mm_set_pd(rhs_d, rhs_d);

    __m128d _dest1;
    __m128d _dest2;

    for (std::size_t i = 0; i < this->size() - 3;
         i += 4, rhs_ptr += 2, ptr += 2, res_ptr+= 2)
    {
        _dest1 = _mm_add_pd(*ptr,  _mm_mul_pd(a, *rhs_ptr ));
        _dest2 = _mm_add_pd(*(ptr + 1), _mm_mul_pd(a, *(rhs_ptr + 1)));

        *res_ptr = _dest1;
        *(res_ptr + 1) = _dest2;
    }
    for (std::size_t i = this->size() % 4; i > 0; --i)
        result[this->size() - i] = this->values[this->size() - i]
            + rhs.values[this->size() - i] * rhs_d;

#else
    for (std::size_t i = 0; i < this->size(); ++i)
        result[i] = this->values[i] + rhs[i] * rhs_d;
#endif
    return result;
}

SSEVector
SSEVector::multiply_sub (smvs::SSEVector const& rhs, double const rhs_d) const
{
    SSEVector result(this->size());
#if ENABLE_SSE_VECTOR && defined(__SSE4_1__)
    __m128d const* ptr = reinterpret_cast<__m128d const*>(this->begin());
    __m128d * res_ptr = reinterpret_cast<__m128d *>(result.begin());
    __m128d const* rhs_ptr =  reinterpret_cast<__m128d const*>(rhs.begin());
    __m128d const a = _mm_set_pd(rhs_d, rhs_d);

    __m128d _dest1;
    __m128d _dest2;

    for (std::size_t i = 0; i < this->size() - 3;
         i += 4, rhs_ptr += 2, ptr += 2, res_ptr+= 2)
    {
        _dest1 = _mm_sub_pd(*ptr,  _mm_mul_pd(a, *rhs_ptr ));
        _dest2 = _mm_sub_pd(*(ptr + 1), _mm_mul_pd(a, *(rhs_ptr + 1)));

        *res_ptr = _dest1;
        *(res_ptr + 1) = _dest2;
    }
    for (std::size_t i = this->size() % 4; i > 0; --i)
        result[this->size() - i] = this->values[this->size() - i]
            - rhs.values[this->size() - i] * rhs_d;
#else
    for (std::size_t i = 0; i < this->size(); ++i)
        result[i] = this->values[i] - rhs[i] * rhs_d;
#endif
    return result;
}

SSEVector &
SSEVector::negate_self (void)
{
    for (std::size_t i = 0; i < this->size(); ++i)
        this->values[i] = -this->values[i];
    return *this;
}

SMVS_NAMESPACE_END
