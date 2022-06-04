#ifndef SPP_BYTE_HPP
#define SPP_BYTE_HPP

#include <cstdint>

#include "operators.hpp"



namespace spp {

	template <uint32_t S>
	struct byte {
		
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
				static_cast<int *>(dst)[i] = LoadOp<int>()(static_cast<int const *>(src) + i);
			}
			
			if constexpr (constexpr uint32_t i = S / 4 * 2; (i + 1) * 2 <= S) {
				static_cast<short *>(dst)[i] = LoadOp<short>()(static_cast<short const *>(src) + i);
			}

			if constexpr (constexpr uint32_t i = S / 2 * 2; i < S) {
				static_cast<char *>(dst)[i] = LoadOp<char>()(static_cast<char const *>(src) + i);
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
				static_cast<int *>(dst)[i] = LoadOp<int>()(static_cast<int const volatile *>(src) + i);
			}
			
			if constexpr (constexpr uint32_t i = S / 4 * 2; (i + 1) * 2 <= S) {
				static_cast<short *>(dst)[i] = LoadOp<short>()(static_cast<short const volatile *>(src) + i);
			}

			if constexpr (constexpr uint32_t i = S / 2 * 2; i < S) {
				static_cast<char *>(dst)[i] = LoadOp<char>()(static_cast<char const volatile *>(src) + i);
			}
		}

		template <template <typename> typename StoreOp = op::stcg, template <typename> typename LoadOp = op::dereference>
		__host__ __device__
		static void copy(void volatile * dst, void const * src) {
			for (uint32_t i = 0; (i + 1) * 16 <= S; ++i) {
				StoreOp<int4>()(static_cast<int4 volatile *>(dst) + i, LoadOp<int4>()(static_cast<int4 const *>(src) + i));
			}
			
			if constexpr (constexpr uint32_t i = S / 16 * 2; (i + 1) * 8 <= S) {
				StoreOp<int2>()(static_cast<int2 volatile *>(dst) + i, LoadOp<int2>()(static_cast<int2 const *>(src) + i));
			}
			
			if constexpr (constexpr uint32_t i = S / 8 * 2; (i + 1) * 4 <= S) {
				StoreOp<int>()(static_cast<int volatile *>(dst) + i, LoadOp<int>()(static_cast<int const *>(src) + i));
			}
			
			if constexpr (constexpr uint32_t i = S / 4 * 2; (i + 1) * 2 <= S) {
				StoreOp<short>()(static_cast<short volatile *>(dst) + i, LoadOp<short>()(static_cast<short const *>(src) + i));
			}

			if constexpr (constexpr uint32_t i = S / 2 * 2; i < S) {
				StoreOp<char>()(static_cast<char volatile *>(dst) + i, LoadOp<char>()(static_cast<char const *>(src) + i));
			}
		}

		byte() = delete;

		~byte() = delete;

		__host__ __device__
		byte(byte const &) = delete;

		__host__ __device__
		byte & operator=(byte const &) = delete;

		__host__ __device__
		byte(byte &&) = delete;

		__host__ __device__
		byte & operator=(byte &&) = delete;

	};

	// template <typename T>
	// using bytes_of	= byte<sizeof(T)>;

	using byte1		= byte<1>;
	using byte2		= byte<2>;
	using byte4		= byte<4>;
	using byte8		= byte<8>;
	using byte16	= byte<16>;



	template <typename T>
	struct bytes_of {

		static constexpr
		int value_alignment{ alignof(T) };

		static constexpr
		int value_bytes{ sizeof(T) };

		__host__ __device__
		static constexpr
		bool aligned_to(int alignment) { return value_alignment % alignment == 0; }



	private:

		template <typename Vector, template <typename> typename StoreOp, template <typename> typename LoadOp>
		__host__ __device__
		static
		void vectorized_copy(void * dst, void const * src) {

			int constexpr this_alignment{ alignof(Vector) };
			int constexpr prev_alignment{ 2 * this_alignment };

			if constexpr (aligned_to(this_alignment)) {

				int constexpr begin{ this_alignment == 16 ? 0 : (
					aligned_to(prev_alignment) ? (value_bytes / prev_alignment * 2) : 0
				) };
				int constexpr end{ value_bytes / this_alignment };

				for (int i = begin; i < end; ++i) {
					StoreOp<Vector>()(
						static_cast<Vector *>(dst) + i,
						LoadOp<Vector>()(static_cast<Vector const *>(src) + i)
					);
				}

			} // ! if

		} // ! vectorized_copy



		template <typename Vector, template <typename> typename StoreOp, template <typename> typename LoadOp>
		__host__ __device__
		static
		void vectorized_copy(void * dst, void const volatile * src) {

			int constexpr this_alignment{ alignof(Vector) };
			int constexpr prev_alignment{ 2 * this_alignment };

			if constexpr (aligned_to(this_alignment)) {

				int constexpr begin{ this_alignment == 16 ? 0 : (
					aligned_to(prev_alignment) ? (value_bytes / prev_alignment * 2) : 0
				) };
				int constexpr end{ value_bytes / this_alignment };

				for (int i = begin; i < end; ++i) {
					StoreOp<Vector>()(
						static_cast<Vector *>(dst) + i,
						LoadOp<Vector>()(static_cast<Vector const volatile *>(src) + i)
					);
				}

			} // ! if

		} // ! vectorized_copy



		template <typename Vector, template <typename> typename StoreOp, template <typename> typename LoadOp>
		__host__ __device__
		static
		void vectorized_copy(void volatile * dst, void const * src) {

			int constexpr this_alignment{ alignof(Vector) };
			int constexpr prev_alignment{ 2 * this_alignment };

			if constexpr (aligned_to(this_alignment)) {

				int constexpr begin{ this_alignment == 16 ? 0 : (
					aligned_to(prev_alignment) ? (value_bytes / prev_alignment * 2) : 0
				) };
				int constexpr end{ value_bytes / this_alignment };

				for (int i = begin; i < end; ++i) {
					StoreOp<Vector>()(
						static_cast<Vector volatile *>(dst) + i,
						LoadOp<Vector>()(static_cast<Vector const *>(src) + i)
					);
				}

			} // ! if

		} // ! vectorized_copy



	public:

		template <template <typename> typename StoreOp = op::assignment, template <typename> typename LoadOp = op::dereference>
		__host__ __device__
		static
		void copy(void * dst, void const * src) {
			vectorized_copy<int4,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<int2,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<int,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<short,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<char,	StoreOp, LoadOp>(dst, src);
		}



		// <todo>
		// Is it valid to use *ldcg*?
		// </todo>
		template <template <typename> typename StoreOp = op::assignment, template <typename> typename LoadOp = op::ldcg>
		__host__ __device__
		static
		void copy(void * dst, void const volatile * src) {
			vectorized_copy<int4,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<int2,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<int,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<short,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<char,	StoreOp, LoadOp>(dst, src);
		}



		// <todo>
		// Is it valid to use *stcg*?
		// </todo>
		template <template <typename> typename StoreOp = op::stcg, template <typename> typename LoadOp = op::dereference>
		__host__ __device__
		static
		void copy(void volatile * dst, void const * src) {
			vectorized_copy<int4,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<int2,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<int,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<short,	StoreOp, LoadOp>(dst, src);
			vectorized_copy<char,	StoreOp, LoadOp>(dst, src);
		}

	};

}



#endif // SPP_BYTE_HPP