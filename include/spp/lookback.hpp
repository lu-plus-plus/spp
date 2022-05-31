#ifndef SPP_LOOKBACK_HPP
#define SPP_LOOKBACK_HPP

#include <type_traits>

#include "types.hpp"
#include "byte.hpp"



namespace spp {

	template <typename T, typename = void>
	struct lookback {

		static_assert(sizeof(T) <= 8, "typename `T` in `lookback<T>` occupies more than 8 bytes.");
		static_assert(std::is_trivial_v<T>, "T must be a trivial type (both default constructible, and copyable, trivially).");

	private:

		enum class e_status : u32 {
			invalid = 0_u32, aggregated = 1_u32, prefixed = 2_u32
		} m_status;

		T m_aggregate;
		T m_prefix;

		constexpr
		__host__ __device__
		lookback(e_status status, T const & aggregation, T const & prefix) : m_status(status), m_aggregate(aggregation), m_prefix(prefix) {}

	public:

		/* constructors */

		constexpr
		lookback() = default;

		/* copy / move constructors */

		constexpr
		lookback(lookback const &) = default;

		constexpr
		lookback(lookback &&) = default;

		// constexpr
		// __host__ __device__
		// lookback(lookback const volatile & rhs) : m_status(rhs.m_status), m_aggregate(rhs.m_aggregate), m_prefix(rhs.m_prefix) {}

		/* assignment opeartors */

		lookback & operator=(lookback const &) = default;
		
		lookback & operator=(lookback &&) = default;

		// __host__ __device__
		// lookback & operator=(lookback const volatile & rhs) {
		// 	// BytesOf<lookback>::copy(this, &rhs);
		// 	this->m_status = rhs.m_status;
		// 	this->m_aggregate = rhs.m_aggregate;
		// 	this->m_prefix = rhs.m_prefix;
			
		// 	return *this;
		// }

		// __device__
		// lookback volatile & operator=(lookback const & rhs) volatile {
		// 	this->m_aggregate = rhs.m_aggregate;
		// 	this->m_prefix = rhs.m_prefix;
		// 	__threadfence();
		// 	this->m_status = rhs.m_status;

		// 	return *this;
		// }

		/* initializers */

		__device__
		lookback volatile & store_invalid() volatile {
			m_status = e_status::invalid;
			m_aggregate = T{};
			m_prefix = T{};

			return *this;
		}

		__device__
		lookback volatile & store_aggregate(T const & value) volatile {
			m_aggregate = value;
			__threadfence();
			m_status = e_status::aggregated;
			
			return *this;
		}

		__device__
		lookback volatile & store_prefix(T const & value) volatile {
			m_prefix = value;
			__threadfence();
			m_status = e_status::prefixed;

			return *this;
		}

		// static constexpr
		// __host__ __device__
		// lookback make_invalid() {
		// 	return lookback{ e_status::invalid, T{}, T{} };
		// }

		// static constexpr
		// __host__ __device__
		// lookback make_aggregate(T value) {
		// 	return lookback{ e_status::aggregated, value, T{} };
		// }

		// static constexpr
		// __host__ __device__
		// lookback make_prefix(T value) {
		// 	return lookback{ e_status::prefixed, T{}, value };
		// }

		/* getters */

		// __host__ __device__
		// bool is_prefixed() const {
		// 	return m_status == e_status::prefixed;
		// }

		// __host__ __device__
		// bool is_aggregated() const {
		// 	return m_status == e_status::aggregated;
		// }

		// __host__ __device__
		// bool is_invalid() const {
		// 	return m_status == e_status::invalid;
		// }

		// __host__ __device__
		// T aggregate() const {
		// 	return m_aggregate;
		// }

		// __host__ __device__
		// T prefix() const {
		// 	return m_prefix;
		// }

		__host__ __device__
		T spin_and_load(bool & is_prefixed) const volatile {
			e_status status;
			do status = m_status; while (status == e_status::invalid);
			is_prefixed = status == e_status::prefixed;
			return is_prefixed ? m_prefix : m_aggregate;

			// lookback result;
			// do result = *this; while (result.is_invalid());
			// return result;
		}

	};



	// template <typename UInt>
	// struct lookback<UInt, std::enable_if_t<std::is_unsigned_v<UInt>>> {

	// private:

	// 	static constexpr
	// 	usize bits = sizeof(UInt) * 8;

	// 	static constexpr
	// 	UInt prefix_bit = UInt(1) << (bits - 1);

	// 	static constexpr
	// 	UInt aggregate_bit = UInt(1) << (bits - 2);

	// 	static constexpr
	// 	UInt value_mask = (~ UInt(0)) >> 2;

	// 	static constexpr
	// 	UInt flags_mask = ~ value_mask;

	// 	UInt data;

	// 	constexpr
	// 	__host__ __device__
	// 	lookback(UInt data_) : data(data_) {}

	// public:

	// 	/* copy / move ctors */

	// 	lookback(lookback const &) noexcept = default;

	// 	lookback(lookback &&) noexcept = default;

	// 	constexpr
	// 	__host__ __device__
	// 	lookback(lookback const volatile & rhs) noexcept : data(rhs.data) {}

	// 	/* copy / move assignment opeartors */

	// 	lookback & operator=(lookback const &) noexcept = default;
		
	// 	lookback & operator=(lookback &&) noexcept = default;

	// 	__host__ __device__
	// 	lookback & operator=(lookback const volatile & rhs) noexcept {
	// 		data = rhs.data;
	// 	}

	// 	__host__ __device__
	// 	lookback volatile & operator=(lookback const & rhs) volatile noexcept {
	// 		data = rhs.data;
	// 	}

	// 	/* initializers */

	// 	static constexpr
	// 	__host__ __device__
	// 	lookback make_invalid() noexcept {
	// 		return lookback{ 0 };
	// 	}

	// 	static constexpr
	// 	__host__ __device__
	// 	lookback make_aggregate(UInt value) noexcept {
	// 		return lookback{ (value & value_mask) | aggregate_bit };
	// 	}

	// 	static constexpr
	// 	__host__ __device__
	// 	lookback make_prefix(UInt value) noexcept {
	// 		return lookback{ (value & value_mask) | prefix_bit };
	// 	}

	// 	/* getters */

	// 	__host__ __device__
	// 	bool is_prefixed() const noexcept {
	// 		return (data & prefix_bit) != 0;
	// 	}

	// 	__host__ __device__
	// 	bool is_aggregated() const noexcept {
	// 		return (data & aggregate_bit) != 0;
	// 	}

	// 	__host__ __device__
	// 	bool is_invalid() const noexcept {
	// 		return (data & flags_mask) == 0;
	// 	}

	// 	__host__ __device__
	// 	UInt value() const noexcept {
	// 		return data & value_mask;
	// 	}

	// 	__host__ __device__
	// 	lookback spin_and_load_value() const volatile noexcept {
	// 		lookback result{ *this };
	// 		while (result.is_invalid()) result = *this;
	// 		return result;
	// 	}
		
	// };

}



#endif // SPP_LOOKBACK_HPP