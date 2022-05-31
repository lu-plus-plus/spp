#ifndef SPP_BYTE_HPP
#define SPP_BYTE_HPP

#include <cstdint>

#include "operators.hpp"



namespace spp {

	template <uint32_t S>
	struct Byte {
		
		struct nothing {} paddings[S];

		template <template <typename> typename LoadOp = op::dereference>
		__host__ __device__
		static void copy(void * dst, void const * src) {
			for (uint32_t i = 0; (i + 1) * 16 <= S; ++i) {
				static_cast<int4 *>(dst)[i] = LoadOp<int4>()(static_cast<int4 const *>(src) + i);
			}
			
			if constexpr (constexpr uint32_t i = S / 16 * 2; (i + 1) * 8 <= S) {
				static_cast<int2 *>(dst)[i] = LoadOp<int2>()(static_cast<int2 const *>(src) + i);
			}
			
			if constexpr (constexpr uint32_t i = S / 8 * 2; (i + 1) * 4 <= S) {
				static_cast<int1 *>(dst)[i] = LoadOp<int1>()(static_cast<int1 const *>(src) + i);
			}
			
			if constexpr (constexpr uint32_t i = S / 4 * 2; (i + 1) * 2 <= S) {
				static_cast<short1 *>(dst)[i] = LoadOp<short1>()(static_cast<short1 const *>(src) + i);
			}

			if constexpr (constexpr uint32_t i = S / 2 * 2; i < S) {
				static_cast<char1 *>(dst)[i] = LoadOp<char1>()(static_cast<char1 const *>(src) + i);
			}
		}

		template <template <typename> typename LoadOp = op::ldcg>
		__host__ __device__
		static void copy(void * dst, void const volatile * src) {
			for (uint32_t i = 0; (i + 1) * 16 <= S; ++i) {
				int4 value{ LoadOp<int4>()(static_cast<int4 const volatile *>(src) + i) };
				static_cast<int4 *>(dst)[i] = value;
			}
			
			if constexpr (constexpr uint32_t i = S / 16 * 2; (i + 1) * 8 <= S) {
				static_cast<int2 *>(dst)[i] = LoadOp<int2>()(static_cast<int2 const volatile *>(src) + i);
			}
			
			if constexpr (constexpr uint32_t i = S / 8 * 2; (i + 1) * 4 <= S) {
				static_cast<int1 *>(dst)[i] = LoadOp<int1>()(static_cast<int1 const volatile *>(src) + i);
			}
			
			if constexpr (constexpr uint32_t i = S / 4 * 2; (i + 1) * 2 <= S) {
				static_cast<short1 *>(dst)[i] = LoadOp<short1>()(static_cast<short1 const volatile *>(src) + i);
			}

			if constexpr (constexpr uint32_t i = S / 2 * 2; i < S) {
				static_cast<char1 *>(dst)[i] = LoadOp<char1>()(static_cast<char1 const volatile *>(src) + i);
			}
		}

		template <template <typename> typename StoreOp = op::stcg, template <typename> typename LoadOp = op::dereference>
		__host__ __device__
		static void copy(void volatile * dst, void const * src) {
			for (uint32_t i = 0; (i + 1) * 16 <= S; ++i) {
				StoreOp()(static_cast<int4 volatile *>(dst) + i, LoadOp<int4>()(static_cast<int4 const *>(src) + i));
			}
			
			if constexpr (constexpr uint32_t i = S / 16 * 2; (i + 1) * 8 <= S) {
				StoreOp()(static_cast<int2 volatile *>(dst) + i, LoadOp<int2>()(static_cast<int2 const *>(src) + i));
			}
			
			if constexpr (constexpr uint32_t i = S / 8 * 2; (i + 1) * 4 <= S) {
				StoreOp()(static_cast<int1 volatile *>(dst) + i, LoadOp<int1>()(static_cast<int1 const *>(src) + i));
			}
			
			if constexpr (constexpr uint32_t i = S / 4 * 2; (i + 1) * 2 <= S) {
				StoreOp()(static_cast<short1 volatile *>(dst) + i, LoadOp<short1>()(static_cast<short1 const *>(src) + i));
			}

			if constexpr (constexpr uint32_t i = S / 2 * 2; i < S) {
				StoreOp()(static_cast<char1 volatile *>(dst) + i, LoadOp<char1>()(static_cast<char1 const *>(src) + i));
			}
		}

		Byte() = delete;

		~Byte() = delete;

		__host__ __device__
		Byte(Byte const &) = delete;

		__host__ __device__
		Byte & operator=(Byte const &) = delete;

		__host__ __device__
		Byte(Byte &&) = delete;

		__host__ __device__
		Byte & operator=(Byte &&) = delete;

	};

	template <typename T>
	using BytesOf = Byte<sizeof(T)>;

	using byte		= Byte<1>;
	using byte2		= Byte<2>;
	using byte4		= Byte<4>;
	using byte8		= Byte<8>;
	using byte16	= Byte<16>;

}



#endif // SPP_BYTE_HPP