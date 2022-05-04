#ifndef SPP_TYPES_HPP
#define SPP_TYPES_HPP

#include <cstdint>
#include <type_traits>



#define $with(...) if (__VA_ARGS__; true)



#define BEGIN_NAMESPACE(ns) namespace ns {
#define END_NAMESPACE(ns) } // ns

#define layoutname template <typename> typename
#define layoutable_typename template <layoutname> typename



BEGIN_NAMESPACE(spp)



template <typename T>
using scalar = T;

template <layoutable_typename Type, layoutname OuterLayout, layoutname InnerLayout>
using AoSoA = OuterLayout<Type<InnerLayout>>;

template <layoutable_typename Type, layoutname Layout>
using AoS = AoSoA<Type, Layout, scalar>;

template <layoutable_typename Type, layoutname Layout>
using SoA = AoSoA<Type, scalar, Layout>;



#define DEFINE_PRIMITIVE(Name, Scalar, Native)	\
template <layoutname Layout = scalar>			\
using Name = Layout<Native>;					\
using Scalar = Name<>

DEFINE_PRIMITIVE(I8,  i8,  int8_t);
DEFINE_PRIMITIVE(I16, i16, int16_t);
DEFINE_PRIMITIVE(I32, i32, int32_t);
DEFINE_PRIMITIVE(I64, i64, int64_t);

DEFINE_PRIMITIVE(U8,  u8,  uint8_t);
DEFINE_PRIMITIVE(U16, u16, uint16_t);
DEFINE_PRIMITIVE(U32, u32, uint32_t);
DEFINE_PRIMITIVE(U64, u64, uint64_t);

DEFINE_PRIMITIVE(F32, f32, float);
DEFINE_PRIMITIVE(F64, f64, double);

#undef DEFINE_PRIMITIVE



template <u64 S>
struct Byte {
	
	struct nothing {} paddings[S];
	
	template <typename T>
	struct deref {
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

	template <template <typename> typename Load = deref>
	__host__ __device__
	static void copy(Byte * dst, Byte const * src) {
		for (u64 i = 0; (i + 1) * 16 <= S; ++i) {
			reinterpret_cast<int4 *>(dst)[i] = Load<int4>()(reinterpret_cast<int4 const *>(src) + i);
		}
		
		if constexpr (constexpr u64 i = S / 16 * 2; (i + 1) * 8 <= S) {
			reinterpret_cast<int2 *>(dst)[i] = Load<int2>()(reinterpret_cast<int2 const *>(src) + i);
		}
		
		if constexpr (constexpr u64 i = S / 8 * 2; (i + 1) * 4 <= S) {
			reinterpret_cast<int1 *>(dst)[i] = Load<int1>()(reinterpret_cast<int1 const *>(src) + i);
		}
		
		if constexpr (constexpr u64 i = S / 4 * 2; (i + 1) * 2 <= S) {
			reinterpret_cast<short1 *>(dst)[i] = Load<short1>()(reinterpret_cast<short1 const *>(src) + i);
		}

		if constexpr (constexpr u64 i = S / 2 * 2; i < S) {
			reinterpret_cast<char1 *>(dst)[i] = Load<char1>()(reinterpret_cast<char1 const *>(src) + i);
		}
	}

	Byte() = default;

	__host__ __device__
	Byte(Byte const & other) : Byte() { copy(this, &other); }
	
	__host__ __device__
	Byte & operator=(Byte const & other) { copy(this, &other); }
	
	__host__ __device__
	Byte(Byte && other) : Byte() { copy(this, &other); }
	
	__host__ __device__
	Byte & operator=(Byte && other) { copy(this, &other); }

	~Byte() = default;

};

template <typename T>
__host__ __device__
Byte<sizeof(T)> const & bytes_of(T const & value) {
	return *reinterpret_cast<Byte<sizeof(T)> const *>(&value);
}

template <typename T>
__host__ __device__
Byte<sizeof(T)> & bytes_of(T & value) {
	return *reinterpret_cast<Byte<sizeof(T)> *>(&value);
}

using byte		= Byte<1>;
using byte2		= Byte<2>;
using byte4		= Byte<4>;
using byte8		= Byte<8>;
using byte16	= Byte<16>;



END_NAMESPACE(spp)



#endif // SPP_TYPES_HPP