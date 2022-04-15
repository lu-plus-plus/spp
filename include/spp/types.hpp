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



END_NAMESPACE(spp)



#endif // SPP_TYPES_HPP