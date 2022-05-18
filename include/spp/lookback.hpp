#ifndef SPP_LOOKBACK_HPP
#define SPP_LOOKBACK_HPP

#include <type_traits>



namespace spp {

	template <typename T, typename = void>
	struct lookback {

		static_assert(sizeof(T) <= 8, "lookback<T> occupies more than 16 bytes.");
		static_assert(std::is_trivially_default_constructible_v<T>, "T must be trivially default-constructible.");
		static_assert(std::is_trivially_copyable_v<T>, "T must be trivally copyable");

	private:

		enum class e_status : u32 {
			invalid = 0_u32, aggregated = 1_u32, prefixed = 2_u32
		} status;

		T value;

		constexpr
		__host__ __device__
		lookback(e_status status_, T const & value_) noexcept :
			status(status_), value(value_) {}

	public:

		/* copy / move ctors */

		lookback(lookback const &) noexcept = default;

		lookback(lookback &&) noexcept = default;

		constexpr
		__host__ __device__
		lookback(lookback const volatile & rhs) noexcept :
			status(rhs.status), value(rhs.value) {}

		/* copy / move assignment opeartors */

		lookback & operator=(lookback const &) noexcept = default;
		
		lookback & operator=(lookback &&) noexcept = default;

		__host__ __device__
		lookback & operator=(lookback const volatile & rhs) noexcept {
			Byte<sizeof(lookback)>::copy(this, &rhs);
		}

		__host__ __device__
		lookback volatile & operator=(lookback const & rhs) volatile noexcept {
			Byte<sizeof(lookback)>::copy(this, &rhs);
		}

		/* initializers */

		static constexpr
		__host__ __device__
		lookback zero() noexcept {
			return lookback(e_status::invalid, T());
		}

		static constexpr
		__host__ __device__
		lookback make_aggregate(T value) noexcept {
			return lookback(e_status::aggregated, value);
		}

		static constexpr
		__host__ __device__
		lookback make_prefix(T value) noexcept {
			return lookback(e_status::prefixed, value);
		}

		/* getters */

		__host__ __device__
		bool is_prefixed() const noexcept {
			return status == e_status::prefixed;
		}

		__host__ __device__
		bool is_aggregated() const noexcept {
			return status == e_status::aggregated;
		}

		__host__ __device__
		bool is_invalid() const noexcept {
			return status == e_status::invalid;
		}

		__host__ __device__
		T get() const noexcept {
			return value;
		}

		__host__ __device__
		lookback wait_and_load() const volatile noexcept {
			lookback result{ *this };
			while (result.is_invalid()) result = *this;
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