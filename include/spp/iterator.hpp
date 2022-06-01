#ifndef SPP_ITERATOR_HPP
#define SPP_ITERATOR_HPP



namespace spp {

	template <typename T>
	struct counting_iterator {
		T value;
		
		__host__ __device__
		counting_iterator(T const & initial) : value(initial) {}

		__host__ __device__
		counting_iterator(T && initial) : value(initial) {}

		__host__ __device__
		T operator*() const {
			return value;
		}
	};

	template <typename T>
	__host__ __device__
	counting_iterator<T> operator+(counting_iterator<T> const & it, uint32_t offset) {
		return counting_iterator{ it.value + offset };
	}

	template <typename T>
	__host__ __device__
	counting_iterator<T> operator+(uint32_t offset, counting_iterator<T> const & it) {
		return counting_iterator{ it.value + offset };
	}

}



#endif // SPP_ITERATOR_HPP