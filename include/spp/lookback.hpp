#ifndef SPP_LOOKBACK_HPP
#define SPP_LOOKBACK_HPP

#include <type_traits>



namespace spp {

	template <typename UInt>
	struct lookback {

		static_assert(std::is_unsigned_v<UInt>, "The underlying type of `lookback` must be unsigned.");

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

		UInt data;

		constexpr
		__host__ __device__
		lookback(UInt data_) : data(data_) {}

	public:

		/* copy / move ctors */

		lookback(lookback const &) noexcept = default;

		lookback(lookback &&) noexcept = default;

		constexpr
		__host__ __device__
		lookback(lookback const volatile & rhs) noexcept : data(rhs.data) {}

		/* copy / move assignment opeartors */

		lookback & operator=(lookback const &) noexcept = default;
		
		lookback & operator=(lookback &&) noexcept = default;

		__host__ __device__
		lookback & operator=(lookback const volatile & rhs) noexcept {
			data = rhs.data;
		}

		__host__ __device__
		lookback volatile & operator=(lookback const & rhs) volatile noexcept {
			data = rhs.data;
		}

		/* initializers */

		static constexpr
		__host__ __device__
		lookback zero() noexcept {
			return lookback{ 0 };
		}

		static constexpr
		__host__ __device__
		lookback make_aggregate(UInt value) noexcept {
			return lookback{ (value & value_mask) | aggregate_bit };
		}

		static constexpr
		__host__ __device__
		lookback make_prefix(UInt value) noexcept {
			return lookback{ (value & value_mask) | prefix_bit };
		}

		/* getters */

		__host__ __device__
		bool is_prefixed() const noexcept {
			return (data & prefix_bit) != 0;
		}

		__host__ __device__
		bool is_aggregated() const noexcept {
			return (data & aggregate_bit) != 0;
		}

		__host__ __device__
		bool is_invalid() const noexcept {
			return (data & flags_mask) == 0;
		}

		__host__ __device__
		UInt get() const noexcept {
			return data & value_mask;
		}

		__host__ __device__
		lookback wait_and_load() const volatile noexcept {
			lookback result{ *this };
			while (result.is_invalid()) result = *this;
			return result;
		}
		
	};

}



#endif // SPP_LOOKBACK_HPP