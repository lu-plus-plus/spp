#ifndef SPP_OPERATORS_MEMORY_HPP
#define SPP_OPERATORS_MEMORY_HPP

#include <utility>



namespace spp::op {

	/* load functions */

	template <typename T>
	struct dereference {
		__host__ __device__
		T & operator()(T * ptr) const {
			return *ptr;
		}

		__host__ __device__
		T const & operator()(T const * ptr) const {
			return *ptr;
		}

		__host__ __device__
		T volatile & operator()(T volatile * ptr) const {
			return *ptr;
		}

		__host__ __device__
		T const volatile & operator()(T const volatile * ptr) const {
			return *ptr;
		}
	};

	template <typename T>
	struct ldg {
		__device__
		T operator()(T const * ptr) const {
			return __ldg(ptr);
		}
	};

	template <typename T>
	struct ldcg {
		__device__
		T operator()(T const volatile * ptr) const {
			return __ldcg(const_cast<T const *>(ptr));
		}
	};



	/* store functions */

	template <typename T>
	struct assignment {
		__host__ __device__
		void operator()(T * ptr, T const & value) const {
			*ptr = value;
		}

		__host__ __device__
		void operator()(T * ptr, T && value) const {
			*ptr = std::move(value);
		}

		__host__ __device__
		void operator()(T volatile * ptr, T const & value) const {
			*ptr = value;
		}

		__host__ __device__
		void operator()(T volatile * ptr, T && value) const {
			*ptr = std::move(value);
		}
	};

	template <typename T>
	struct stcg {
		__device__
		void operator()(T * ptr, T const & value) const {
			__stcg(ptr, value);
		}

		__device__
		void operator()(T volatile * ptr, T const & value) const {
			__stcg(const_cast<T *>(ptr), value);
		}

		// <comment>
		// no operator()(T *, T &&): __stcg has no overloading for rvalue
		// </comment>
	};

}



#endif // SPP_OPERATORS_MEMORY_HPP