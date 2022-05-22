#ifndef SPP_IDENTITY_HPP
#define SPP_IDENTITY_HPP



namespace spp::op {

	template <typename T>
	struct identity_element {
		__host__ __device__
		T operator()() const {
			return T(0);
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
	};

	template <>
	struct identity_function<void> {
		template <typename T>
		__host__ __device__
		T operator()(T const & value) const {
			return value;
		}
	};

} // namespace spp::op



#endif // SPP_IDENTITY_HPP