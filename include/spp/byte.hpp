#ifndef SPP_BYTE_HPP
#define SPP_BYTE_HPP

#include "traits.hpp"
#include "operators/memory.hpp"



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

	using byte1		= byte<1>;
	using byte2		= byte<2>;
	using byte4		= byte<4>;
	using byte8		= byte<8>;
	using byte16	= byte<16>;



	template <typename T>
	struct bytes_of {

	private:

		static constexpr
		int align_in_bytes{ alignof(T) };

		static constexpr
		int size_in_bytes{ sizeof(T) };

		__host__ __device__
		static constexpr
		bool aligned_to(int alignment) { return align_in_bytes % alignment == 0; }



		template <typename Vector, template <typename> typename StoreOp, template <typename> typename LoadOp, typename DstPtr, typename SrcPtr>
		__host__ __device__
		static
		void vectorized_copy(DstPtr dst, SrcPtr src) {

			int constexpr this_align{ alignof(Vector) };
			int constexpr prev_align{ 2 * this_align };

			if constexpr (aligned_to(this_align)) {

				int constexpr begin{ this_align == 16 ? 0 : (
					aligned_to(prev_align) ? (size_in_bytes / prev_align * 2) : 0
				) };
				int constexpr end{ size_in_bytes / this_align };

				for (int i = begin; i < end; ++i) {
					StoreOp<Vector>()(
						pointed_cast<DstPtr, Vector>(dst) + i,
						LoadOp<Vector>()(pointed_cast<SrcPtr, Vector>(src) + i)
					);
				}

			}

		}



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