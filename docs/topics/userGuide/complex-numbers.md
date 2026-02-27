# Complex numbers

<!---IMPORT samples.docs.userGuide.ComplexNumbers-->

<web-summary>
Learn how to work with complex numbers in Multik using ComplexFloat and ComplexDouble.
Create complex ndarrays, perform arithmetic, and use math/stat functions on complex data.
</web-summary>

<card-summary>
Use ComplexFloat and ComplexDouble in ndarrays.
Create, cast, and compute with complex-valued data in Multik.
</card-summary>

<link-summary>
Work with complex numbers and complex ndarrays in Multik.
</link-summary>

## Overview

Multik supports complex-valued data through two element types:

* `ComplexFloat` — single-precision complex numbers
* `ComplexDouble` — double-precision complex numbers

These types behave like regular numeric values with `re` (real) and `im` (imaginary) parts,
and they can be stored in ndarrays just like other primitive numeric types.

## Creating complex values

<!---FUN creating_complex_values-->

```kotlin
val z1 = ComplexFloat(1f, -2f)
val z2 = ComplexDouble(3.0, 4.0)

val i1 = 1f.i      // 0 + 1i (ComplexFloat)
val i2 = 2.0.i     // 0 + 2i (ComplexDouble)

val real = 5.0.toComplexDouble() // 5 + 0i
```

<!---END-->

`ComplexFloat` and `ComplexDouble` provide useful helpers:

<!---FUN complex_helpers-->

```kotlin
val z = ComplexDouble(1.0, -2.0)
val conj = z.conjugate()
val magnitude = z.abs()
val angle = z.angle()
```

<!---END-->

## Creating complex ndarrays

You can build complex arrays using literals or dedicated complex array builders.

<!---FUN creating_complex_ndarrays-->

```kotlin
val a = mk.ndarray(
    mk[
        ComplexDouble(1.0, 2.0),
        ComplexDouble(3.0, -1.0)
    ]
)

val b = mk.ones<ComplexFloat>(2, 2)
```

<!---END-->

From complex primitive arrays:

<!---FUN from_complex_primitive_arrays-->

```kotlin
val data = complexDoubleArrayOf(1.0 + 2.0.i, 3.0 + 4.0.i, 5.0 + 6.0.i)
val c = mk.ndarray(data)
```

<!---END-->

You can also generate complex grids with index-based factories:

<!---FUN complex_grid_factory-->

```kotlin
val grid = mk.d2arrayIndices(2, 3) { i, j -> ComplexFloat(i.toFloat(), j.toFloat()) }
```

<!---END-->

## Arithmetic with complex arrays

Complex ndarrays support the usual element-wise operations:

<!---FUN complex_arithmetic-->

```kotlin
val x = mk.ndarray(mk[ComplexDouble(1.0, 1.0), ComplexDouble(2.0, -1.0)])
val y = mk.ndarray(mk[ComplexDouble(0.5, 0.0), ComplexDouble(1.0, 2.0)])

val sum = x + y
val product = x * y
```

<!---END-->

When adding or multiplying by scalars, use complex scalars:

<!---FUN complex_scalar_operations-->

```kotlin
val scaled = x + 2.0.toComplexDouble()
```

<!---END-->

## Math functions

Many math functions work with complex arrays. For example:

```kotlin
val w = x.exp()
val s = x.sin()
val magnitudes = x.map { it.abs() }
```

## Converting between real and complex types

You can cast a real array to a complex type with `toType`,
which sets the imaginary part to zero:

<!---FUN real_to_complex_conversion-->

```kotlin
val real = mk.ndarray(mk[1.0, 2.0, 3.0])
val complex = real.toType<Double, ComplexDouble, D1>()
```

<!---END-->

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="type-casting.md" summary="Convert ndarrays between numeric and complex dtypes."/>
  <a href="creating-multidimensional-arrays.md" summary="Build ndarrays from literals, collections, primitives, and factories."/>
</category>
<category ref="api-docs">
  <a href="mathematical.md" summary="Math functions available for real and complex ndarrays.">Mathematical Functions</a>
</category>
</seealso>
