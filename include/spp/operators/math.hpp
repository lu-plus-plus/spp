#ifndef SPP_OPERATORS_MATH_HPP
#define SPP_OPERATORS_MATH_HPP



namespace spp::op {

	template <typename T>
	struct identity_element {
		__host__ __device__
		T operator()() const {
			return T{ 0 };
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

} // namespace spp::op



#endif // SPP_OPERATORS_MATH_HPP