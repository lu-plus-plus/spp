#ifndef SPP_BYTE_HPP
#define SPP_BYTE_HPP

#include <cstdint>

#include "operators.hpp"



namespace spp {

	template <uint32_t S>
	struct byte {
		
		struct nothing {} paddings[S];

		byte() = default;

		~byte() = default;

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