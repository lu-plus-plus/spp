#ifndef SPP_IDENTITY_HPP
#define SPP_IDENTITY_HPP

#include <type_traits>

#include "types.hpp"



namespace spp::op {

	template <typename T, typename = void>
	struct identity;

	template <typename T>
	struct identity<T, std::enable_if_t<std::is_integral_v<T> or std::is_floating_point_v<T>>> {
		__host__ __device__
		T operator()() const noexcept {
			return T(0);
		}
	};

} // namespace spp::op



#endif // SPP_IDENTITY_HPP