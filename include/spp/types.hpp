#ifndef SPP_TYPES_HPP
#define SPP_TYPES_HPP

#include <cstdint>



#define with_(...) if (__VA_ARGS__; true)



namespace spp {

	using i8	= int8_t;
	using i16	= int16_t;
	using i32	= int32_t;
	using i64	= int64_t;

	using u8	= uint8_t;
	using u16	= uint16_t;
	using u32	= uint32_t;
	using u64	= uint64_t;

	using f32	= float;
	using f64	= double;



	namespace op {

		template <typename T>
		struct dereference {
			__host__ __device__
			T operator()(T const * ptr) const noexcept {
				return *ptr;
			}
		};

		template <typename T>
		struct ldg {
			__host__ __device__
			T operator()(T const * ptr) const noexcept {
				return __ldg(ptr);
			}
		};

	}



	template <u32 S>
	struct Byte {
		
		struct nothing {} paddings[S];

		template <template <typename> typename LoadOp = op::dereference>
		__host__ __device__
		static void copy(void * dst, void const * src) noexcept {
			for (u32 i = 0; (i + 1) * 16 <= S; ++i) {
				static_cast<int4 *>(dst)[i] = LoadOp<int4>()(static_cast<int4 const *>(src) + i);
			}
			
			if constexpr (constexpr u32 i = S / 16 * 2; (i + 1) * 8 <= S) {
				static_cast<int2 *>(dst)[i] = LoadOp<int2>()(static_cast<int2 const *>(src) + i);
			}
			
			if constexpr (constexpr u32 i = S / 8 * 2; (i + 1) * 4 <= S) {
				static_cast<int1 *>(dst)[i] = LoadOp<int1>()(static_cast<int1 const *>(src) + i);
			}
			
			if constexpr (constexpr u32 i = S / 4 * 2; (i + 1) * 2 <= S) {
				static_cast<short1 *>(dst)[i] = LoadOp<short1>()(static_cast<short1 const *>(src) + i);
			}

			if constexpr (constexpr u32 i = S / 2 * 2; i < S) {
				static_cast<char1 *>(dst)[i] = LoadOp<char1>()(static_cast<char1 const *>(src) + i);
			}
		}

		Byte() = default;

		~Byte() = default;

		__host__ __device__
		Byte(Byte const &) = delete;

		__host__ __device__
		Byte & operator=(Byte const &) = delete;

		__host__ __device__
		Byte(Byte &&) = delete;

		__host__ __device__
		Byte & operator=(Byte &&) = delete;

	};

	using byte		= Byte<1>;
	using byte2		= Byte<2>;
	using byte4		= Byte<4>;
	using byte8		= Byte<8>;
	using byte16	= Byte<16>;



	// template <typename T>
	// __host__ __device__
	// Byte<sizeof(T)> const & bytes_of(T const & value) {
	// 	return *reinterpret_cast<Byte<sizeof(T)> const *>(&value);
	// }

	// template <typename T>
	// __host__ __device__
	// Byte<sizeof(T)> & bytes_of(T & value) {
	// 	return *reinterpret_cast<Byte<sizeof(T)> *>(&value);
	// }

	// template <typename T>
	// __host__ __device__
	// auto bytes_at(T const * ptr) {
	// 	return reinterpret_cast<Byte<sizeof(T)> const *>(ptr);
	// }

	// template <typename T>
	// __host__ __device__
	// auto bytes_at(T * ptr) {
	// 	return reinterpret_cast<Byte<sizeof(T)> *>(ptr);
	// }

} // namespace spp



#endif // SPP_TYPES_HPP