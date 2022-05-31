#ifndef SPP_DEVICE_PTR_HPP
#define SPP_DEVICE_PTR_HPP

#include "kernel_launch.hpp"



namespace spp {

	template <typename T>
	class device_ptr {

		T * p_data;
		std::size_t * p_ref_count;

		void inc() const {
			if (p_ref_count) ++*p_ref_count;
		}

		void dec() const {
			if (p_ref_count and --*p_ref_count == 0) {
				cudaCheck(cudaFree(p_data));
			}
		}

	public:

		device_ptr() : p_data(nullptr), p_ref_count(nullptr) {}

		device_ptr(T * ptr) : p_data(ptr), p_ref_count(new std::size_t(1)) {}
		
		device_ptr(device_ptr const & other) : p_data(other.p_data), p_ref_count(other.p_ref_count) {
			other.inc();
		}

		device_ptr(device_ptr && other) : p_data(other.p_data), p_ref_count(other.p_ref_count) {
			other.p_data = nullptr;
			other.p_ref_count = nullptr;
		}

		device_ptr & operator=(device_ptr const & other) {
			this->dec();
			this->p_data = other.p_data;
			this->p_ref_count = other.p_ref_count;
			
			other.inc();
			
			return *this;
		}

		device_ptr & operator=(device_ptr && other) {
			this->dec();
			this->p_data = other.p_data;
			this->p_ref_count = other.p_ref_count;
			
			other.p_data = nullptr;
			other.p_ref_count = nullptr;

			return *this;
		}

		~device_ptr() {
			this->dec();
		}

		T * get() const {
			return p_data ? p_data : nullptr;
		}

		operator T * () const {
			return get();
		}

	};

	template <typename T>
	device_ptr<T> device_alloc(u32 size_in_items) {
		T * ptr = nullptr;
		cudaCheck(cudaMalloc(&ptr, sizeof(T) * size_in_items));
		return device_ptr<T>(ptr);
	}

	template <typename T>
	device_ptr<T> device_copy_from(T const * begin, T const * end) {
		u32 const size_in_items = end - begin;
		auto ptr = device_alloc<T>(size_in_items);
		cudaCheck(cudaMemcpy(ptr.get(), begin, sizeof(T) * size_in_items, cudaMemcpyHostToDevice));
		return ptr;
	}

}



#endif // SPP_DEVICE_PTR_HPP