#ifndef SPP_ITERATOR_HPP
#define SPP_ITERATOR_HPP

#include <iterator>



namespace spp {

	template <typename T>
	struct counting_iterator {
		T value;
		
		using iterator_category = std::input_iterator_tag;
		using value_type = T;
		using difference_type = uint32_t;
		using pointer = T const *;
		using reference = T;

		__host__ __device__
		counting_iterator(T const & initial) : value(initial) {}

		__host__ __device__
		counting_iterator(T && initial) : value(initial) {}

		__host__ __device__
		T operator*() const {
			return value;
		}

		__host__ __device__
		counting_iterator & operator++() {
			++value;
			return *this;
		}

		__host__ __device__
		counting_iterator operator++(int) const {
			return counting_iterator{ ++value };
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

	template <typename T>
	__host__ __device__
	bool operator==(counting_iterator<T> const & a, counting_iterator<T> const & b) {
		return a.value == b.value;
	}

	template <typename T>
	__host__ __device__
	bool operator!=(counting_iterator<T> const & a, counting_iterator<T> const & b) {
		return a.value != b.value;
	}

}



#endif // SPP_ITERATOR_HPP