#ifndef SPP_MEMORY_SECTIONS_HPP
#define SPP_MEMORY_SECTIONS_HPP

#include <utility>

#include "kernel_launch.hpp"



namespace spp {

	template <typename ... Args>
	struct memory_sections {

	private:

		void * m_ptr;
		uint32_t m_offsets[sizeof...(Args)];
		uint32_t m_size;

		static
		uint32_t exclusive_prefix(uint32_t * data, uint32_t size) {
			uint32_t partial_result{ 0u };
			for (uint32_t i = 0; i < size; ++i) {
				uint32_t const temp{ data[i] };
				data[i] = partial_result;
				partial_result += temp;
			}
			return partial_result;
		}

		template <uint32_t I, typename ... Types>
		struct i_th_element;

		template <uint32_t I, typename Head, typename ... Tail>
		struct i_th_element<I, Head, Tail ...> {
			using type = typename i_th_element<I - 1, Tail ...>::type;
		};

		template <typename Head, typename ... Tail>
		struct i_th_element<0, Head, Tail ...> {
			using type = Head;
		};

		template <uint32_t I, typename ... Types>
		using i_th_element_t = typename i_th_element<I, Args ...>::type;

		template <uint32_t ... UInts>
		auto make_ptrs(std::integer_sequence<uint32_t, UInts ...>) const {
			return std::tuple<Args * ...>{ static_cast<Args *>(static_cast<void *>(static_cast<std::byte *>(m_ptr) + m_offsets[UInts]))... };
		}

	public:

		memory_sections(void * ptr, std::initializer_list<uint32_t> sizes) : m_ptr{ ptr }, m_offsets{}, m_size{} {
			uint32_t sizeof_args[sizeof...(Args)] = { static_cast<uint32_t>(sizeof(Args))... };
			
			for (auto it = sizes.begin(); it != sizes.end(); ++it) {
				uint32_t const i = it - sizes.begin();
				m_offsets[i] = aligned_bytes_for(sizeof_args[i] * (*it));
			}

			m_size = exclusive_prefix(m_offsets, sizes.size());
		}

		template <uint32_t I>
		auto ptr() const {
			return static_cast<i_th_element_t<I, Args ...> *>(
				static_cast<void *>(static_cast<std::byte *>(m_ptr) + m_offsets[I])
			);
		}

		std::tuple<Args * ...> ptrs() const {
			return make_ptrs(std::make_integer_sequence<uint32_t, sizeof...(Args)>());
		}

		uint32_t size() const {
			return m_size;
		}

	};

}



#endif // SPP_MEMORY_SECTIONS_HPP