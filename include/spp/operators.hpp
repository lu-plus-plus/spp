#ifndef SPP_OPERATORS_HPP
#define SPP_OPERATORS_HPP



namespace spp::op {

	template <typename T>
	struct identity_element {
		__host__ __device__
		T operator()() const {
			return T{ 0 };
		}
	};



	template <typename T = void>
	struct plus {
		__host__ __device__
		T operator()(T const & a, T const & b) const {
			return a + b;
		}
	};

	template <>
	struct plus<void> {
		template <typename T>
		__host__ __device__
		T operator()(T const & a, T const & b) const {
			return a + b;
		}
	};



	template <typename T = void>
	struct identity_function {
		__host__ __device__
		T operator()(T const & value) const {
			return value;
		}

		__host__ __device__
		T operator()(T const & value, uint32_t index) const {
			return value;
		}
	};

	template <>
	struct identity_function<void> {
		template <typename T>
		__host__ __device__
		T operator()(T const & value) const {
			return value;
		}

		template <typename T>
		__host__ __device__
		T operator()(T const & value, uint32_t index) const {
			return value;
		}		
	};



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
			*ptr = value;
		}

		__host__ __device__
		void operator()(T volatile * ptr, T const & value) const {
			*ptr = value;
		}

		__host__ __device__
		void operator()(T volatile * ptr, T && value) const {
			*ptr = value;
		}
	};

	template <typename T>
	struct stcg {
		__device__
		void operator()(T volatile * ptr, T const & value) const {
			__stcg(const_cast<T *>(ptr), value);
		}

		__device__
		void operator()(T volatile * ptr, T && value) const {
			__stcg(const_cast<T *>(ptr), value);
		}
	};

} // namespace spp::op



#endif // SPP_OPERATORS_HPP