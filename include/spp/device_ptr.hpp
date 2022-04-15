#ifndef SPP_DEVICE_PTR_HPP
#define SPP_DEVICE_PTR_HPP

#include "types.hpp"
#include "log.hpp"



namespace spp {

	template <typename T>
	class device_ptr {

		struct details_t {

			T * ptr;
			u32 ref_count;

			void inc() { ++ref_count; }
			void dec() { if (--ref_count == 0) cudaCheck(cudaFree(ptr)); }

			details_t(T * ptr) : ptr(ptr), ref_count(1) {}

		};

		details_t * p_details;

	public:

		device_ptr() : p_details(nullptr) {}

		device_ptr(T * ptr) : p_details(new details_t(ptr)) {}
		
		device_ptr(device_ptr const & other) : p_details(other.p_details) {
			other.p_details->inc();
		}

		device_ptr & operator=(device_ptr const & other) {
			this->p_details->dec();
			this->p_details = other.p_details;
			
			other.p_details->inc();
			
			return *this;
		}

		device_ptr(device_ptr && other) : p_details(other.p_details) {
			other.p_details = nullptr;
		}

		device_ptr & operator=(device_ptr && other) {
			this->p_details->dec();
			this->p_details = other.p_details;
			
			other.p_details = nullptr;

			return *this;
		}

		~device_ptr() {
			this->p_details->dec();
		}

		T * get() const {
			return p_details ? p_details->ptr : nullptr;
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
	device_ptr<T> device_alloc(T const * begin, T const * end) {
		u32 const size_in_items = end - begin;
		auto ptr = device_alloc<T>(size_in_items);
		cudaCheck(cudaMemcpy(ptr.get(), begin, sizeof(T) * size_in_items, cudaMemcpyHostToDevice));
		return ptr;
	}

}



#endif // SPP_DEVICE_PTR_HPP