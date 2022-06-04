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

		/* constructors and assignment operators */

		lookback() = default;

		constexpr
		lookback(lookback const &) = default;

		constexpr
		lookback(lookback &&) = default;

		lookback & operator=(lookback const &) = default;
		
		lookback & operator=(lookback &&) = default;

		/* setters for struct in global memory */

		__device__
		lookback volatile & store_invalid() volatile {
			// m_status = e_status::invalid;
			// m_aggregate = T{};
			// m_prefix = T{};

			e_status const status{ e_status::invalid };
			T const value{};

			bytes_of<e_status>::copy(&m_status, &status);
			bytes_of<T>::copy(&m_aggregate, &value);
			bytes_of<T>::copy(&m_prefix, &value);

			return *this;
		}

		__device__
		lookback volatile & store_aggregate(T const & value) volatile {
			// m_aggregate = value;
			// __threadfence();
			// m_status = e_status::aggregated;

			bytes_of<T>::copy(&m_aggregate, &value);
			__threadfence();
			e_status const status{ e_status::aggregated };
			bytes_of<e_status>::copy(&m_status, &status);

			return *this;
		}

		__device__
		lookback volatile & store_prefix(T const & value) volatile {
			// m_prefix = value;
			// __threadfence();
			// m_status = e_status::prefixed;

			bytes_of<T>::copy(&m_prefix, &value);
			__threadfence();
			e_status const status{ e_status::prefixed };
			bytes_of<e_status>::copy(&m_status, &status);

			return *this;
		}

		/* getters for struct in global memory */

		__device__
		T spin_and_load(bool & is_prefixed) const volatile {
			e_status status;
			do status = m_status; while (status == e_status::invalid);
			is_prefixed = status == e_status::prefixed;

			// return is_prefixed ? m_prefix : m_aggregate;

			T result;
			bytes_of<T>::copy(&result, &(is_prefixed ? m_prefix : m_aggregate));
			return result;
		}

	};



	template <typename UInt>
	struct lookback<UInt, std::enable_if_t<std::is_unsigned_v<UInt>>> {

	private:

		static constexpr
		usize bits = sizeof(UInt) * 8;

		static constexpr
		UInt prefix_bit = UInt(1) << (bits - 1);

		static constexpr
		UInt aggregate_bit = UInt(1) << (bits - 2);

		static constexpr
		UInt value_mask = (~ UInt(0)) >> 2;

		static constexpr
		UInt flags_mask = ~ value_mask;

		UInt m_data;

		constexpr
		__host__ __device__
		lookback(UInt value) : m_data(value) {}

	public:

		/* constructors and assignment operators */

		lookback() = default;

		constexpr
		lookback(lookback const &) = default;

		constexpr
		lookback(lookback &&) = default;

		lookback & operator=(lookback const &) = default;
		
		lookback & operator=(lookback &&) = default;

		/* setters for struct in global memory */

		__host__ __device__
		lookback volatile & store_invalid() volatile {
			m_data = UInt(0);
			return *this;
		}

		__host__ __device__
		lookback volatile & store_aggregate(UInt value) volatile {
			m_data = (value & value_mask) | aggregate_bit;
			return *this;
		}

		__host__ __device__
		lookback volatile & store_prefix(UInt value) volatile {
			m_data = (value & value_mask) | prefix_bit;
			return *this;
		}

		/* getters for struct in global memory */

		static
		__host__ __device__
		bool is_prefixed(UInt data) {
			return (data & prefix_bit) != 0;
		}

		static
		__host__ __device__
		bool is_aggregated(UInt data) {
			return (data & aggregate_bit) != 0;
		}

		static
		__host__ __device__
		bool is_invalid(UInt data) {
			return (data & flags_mask) == 0;
		}

		static
		__host__ __device__
		UInt value(UInt data) {
			return data & value_mask;
		}

		__host__ __device__
		UInt spin_and_load(bool & prefixed) const volatile {
			UInt data;
			do data = m_data; while (is_invalid(data));
			
			prefixed = is_prefixed(data);
			
			return value(data);
		}
		
	};

}



#endif // SPP_LOOKBACK_HPP