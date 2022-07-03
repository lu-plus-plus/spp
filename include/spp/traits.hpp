#ifndef SPP_TRAITS_HPP
#define SPP_TRAITS_HPP

#include <type_traits>



namespace spp {

	template <typename T>
	using dereference_t = decltype(*std::declval<T>());

	template <typename Fn, typename ... Args>
	using apply_t = decltype(std::declval<Fn>()(std::declval<Args>()...));



	// struct pointed_cast {

	// 	template <typename From, typename To>
	// 	auto operator()(From *) const {
	// 		return static_cast<To *>(nullptr);
	// 	}

	// 	template <typename From, typename To>
	// 	auto operator()(From const *) const {
	// 		return static_cast<To const *>(nullptr);
	// 	}

	// 	template <typename From, typename To>
	// 	auto operator()(From volatile *) const {
	// 		return static_cast<To volatile *>(nullptr);
	// 	}

	// 	template <typename From, typename To>
	// 	auto operator()(From const volatile *) const {
	// 		return static_cast<To const volatile *>(nullptr);
	// 	}

	// };

	namespace impl {

		template <typename FromPtr, typename To>
		struct pointed_cast;

		template <typename T, typename To>
		struct pointed_cast<T *, To> {
			using type = To *;
		};

		template <typename T, typename To>
		struct pointed_cast<T const *, To> {
			using type = To const *;
		};

		template <typename T, typename To>
		struct pointed_cast<T volatile *, To> {
			using type = To volatile *;
		};

		template <typename T, typename To>
		struct pointed_cast<T const volatile *, To> {
			using type = To const volatile *;
		};

	}

	template <typename FromPtr, typename To>
	using pointed_cast = typename impl::pointed_cast<FromPtr, To>::type;

}



#endif // SPP_TRAITS_HPP