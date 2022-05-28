#ifndef SPP_TRAITS_HPP
#define SPP_TRAITS_HPP

#include <type_traits>



namespace spp {

	template <typename T>
	using dereference_t = decltype(*std::declval<T>());

	template <typename Fn, typename ... Args>
	using apply_t = decltype(std::declval<Fn>()(std::declval<Args>()...));

}



#endif // SPP_TRAITS_HPP