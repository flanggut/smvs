/*
 * Copyright (c) 2016, Fabian Langguth
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD 3-Clause license. See the LICENSE.txt file for details.
 */

#ifndef SMVS_SSE_VECTOR_HEADER
#define SMVS_SSE_VECTOR_HEADER

#include <cmath>

#include "util/aligned_memory.h"

#include "defines.h"

SMVS_NAMESPACE_BEGIN

class SSEVector
{
public:
    SSEVector (void) = default;
    SSEVector (std::size_t size, double const& value = 0.0);
    void resize (std::size_t size, double const& value = 0.0);
    void fill (double const& value = 0.0);
    void clear (void);
    std::size_t size (void) const;

    double const& operator[] (std::size_t index) const;
    double & operator[] (std::size_t index);
    double const& at (std::size_t index) const;
    double & at (std::size_t index);


    double const* begin (void) const;
    double * begin (void);
    double const* end (void) const;
    double * end (void);

    double norm (void) const;
    double square_norm (void) const;
    double dot (SSEVector const& rhs) const;

    SSEVector & negate_self (void);

    SSEVector multiply (double const rhs) const;
    SSEVector add (SSEVector const& rhs) const;
    SSEVector subtract (SSEVector const& rhs) const;

private:
    util::AlignedMemory<double> values;
};


inline
SSEVector::SSEVector (std::size_t size, double const& value)
{
    this->resize(size, value);
}

inline void
SSEVector::resize (std::size_t size, double const& value)
{
    this->values.resize(size, value);
}

inline void
SSEVector::fill (double const& value)
{
    std::fill(this->values.begin(), this->values.end(), value);
}


inline void
SSEVector::clear (void)
{
    this->values = util::AlignedMemory<double>(0);
}

inline std::size_t
SSEVector::size (void) const
{
    return this->values.size();
}

inline double const&
SSEVector::operator[] (std::size_t index) const
{
    return this->values[index];
}

inline double &
SSEVector::operator[] (std::size_t index)
{
    return this->values[index];
}

inline double const&
SSEVector::at (std::size_t index) const
{
    return this->values[index];
}

inline double &
SSEVector::at (std::size_t index)
{
    return this->values[index];
}


inline double const*
SSEVector::begin (void) const
{
    return this->values.data();
}

inline double *
SSEVector::begin (void)
{
    return this->values.data();
}

inline double const*
SSEVector::end (void) const
{
    return this->values.data() + this->values.size();
}

inline double *
SSEVector::end (void)
{
    return this->values.data() + this->values.size();
}

inline double
SSEVector::norm (void) const
{
    return std::sqrt(this->dot(*this));
}

inline double
SSEVector::square_norm (void) const
{
    return this->dot(*this);
}

SMVS_NAMESPACE_END

#endif /* SSE_VECTOR_HEADER */
