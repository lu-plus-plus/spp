#ifndef SPP_DEVICE_VECTOR_HPP
#define SPP_DEVICE_VECTOR_HPP

#include <vector>

#include "log.hpp"



namespace spp {

	template <typename T>
	class device_vector {

		T *				m_data;
		std::size_t		m_size;
		std::size_t		m_capacity;

		static constexpr
		std::size_t default_capacity = 8;

		static
		T * alloc_items(std::size_t items) {
			T * ptr = nullptr;
			cudaCheck(cudaMalloc(&ptr, sizeof(T) * items));
			return ptr;
		}

		static
		std::size_t inclusive_next_power_of_2(std::size_t i) {
			if (i <= 1) return 1;
			--i;
			return (1 << (32 - __builtin_clzl(i)));
		}

	public:

		device_vector() : m_data(alloc_items(default_capacity)), m_size(0), m_capacity(default_capacity) {}

		device_vector(std::vector<T> const & host_vector) :
			m_data(alloc_items(host_vector.capacity())),
			m_size(host_vector.size()),
			m_capacity(host_vector.capacity()) {

			cudaCheck(cudaMemcpy(m_data, host_vector.data(), sizeof(T) * m_size, cudaMemcpyHostToDevice));
		}
		
		device_vector(std::size_t size) :
			m_data(alloc_items(inclusive_next_power_of_2(size))),
			m_size(size),
			m_capacity(inclusive_next_power_of_2(size)) {}

		device_vector(std::size_t size, T const & init) :
			m_data(alloc_items(inclusive_next_power_of_2(size))),
			m_size(size),
			m_capacity(inclusive_next_power_of_2(size)) {

			std::vector<T> host_vector(m_size, init);
			cudaCheck(cudaMemcpy(m_data, host_vector.data(), sizeof(T) * m_size, cudaMemcpyHostToDevice));
		}

		~device_vector() {
			cudaCheck(cudaFree(m_data));
		}

		T * data() const {
			return m_data;
		}

		std::size_t size() const {
			return m_size;
		}

		std::size_t capacity() const {
			return m_capacity;
		}

		std::vector<T> to_host() const {
			std::vector<T> host_vector(m_size);
			cudaCheck(cudaMemcpy(host_vector.data(), m_data, sizeof(T) * m_size, cudaMemcpyDeviceToHost));
			return host_vector;
		}

	};

}



#endif // SPP_DEVICE_VECTOR_HPP