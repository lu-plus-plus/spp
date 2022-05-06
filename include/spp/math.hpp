#ifndef SPP_UTILS_HPP
#define SPP_UTILS_HPP

#include "types.hpp"



namespace spp {

	__host__ __device__
	constexpr u32 ceiled_div(u32 dividend, u32 divisor) {
		return (dividend + divisor - 1) / divisor;
	}

}



#endif // SPP_UTILS_HPP