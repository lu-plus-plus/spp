#ifndef SPP_RNG_HPP
#define SPP_RNG_HPP

#include <ctime>
#include <random>
#include <algorithm>
#include <type_traits>

#include "device_vector.hpp"



namespace spp {

	template <typename T>
	struct rng {

		template <typename U>
		using distribution_template = std::conditional_t<
			std::is_integral_v<T>,
			std::uniform_int_distribution<U>,
			std::uniform_real_distribution<U>
		>;

		std::default_random_engine random_generator;
		distribution_template<T> distribution;

		rng(T lower, T upper) : random_generator(std::default_random_engine{ static_cast<unsigned long>(std::time(nullptr)) }()), distribution(lower, upper) {}

		T num() {
			return T(distribution(random_generator));
		}

		device_vector<T> seq(size_t items) {
			std::vector<T> h_data(items);
			std::generate(h_data.begin(), h_data.end(), [this] () -> T { return num(); });
			return device_vector<T>{ h_data };
		}
	};

}



#endif // SPP_RNG_HPP