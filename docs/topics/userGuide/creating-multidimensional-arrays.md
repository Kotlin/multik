# Creating multidimensional arrays

<!---IMPORT samples.docs.userGuide.CreatingMultidimensionalArrays-->

<web-summary>
Learn the main ways to create ndarrays in Multik, from literal-style construction to factory functions.
See how to control shape, data type, and dimension in a clear, Kotlin-friendly API.
</web-summary>

<card-summary>
Create ndarrays from literals, collections, and factory functions.
Control shape, dtype, and dimension with explicit and inferred options.
</card-summary>

<link-summary>
Explore the primary ndarray creation patterns in Multik: literals, collections, primitives, and factories.
</link-summary>

## Overview

Multik offers multiple ways to build ndarrays, depending on how much control you need.
You can rely on type and dimension inference for quick experiments, or specify dtype and shape explicitly
for production-grade, predictable behavior.

## Literal-style construction with `mk[]`

The fastest way to create an ndarray is to use the `mk[]` literal builder and pass it to `mk.ndarray`.
The element type and dimension are inferred from the nested structure.

<!---FUN literal_construction-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
// [1, 2, 3]

val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
/*
[[1.5, 2.1, 3.0],
 [4.0, 5.0, 6.0]]
 */
```

<!---END-->

When you need a specific dimension type, you can state it explicitly:

<!---FUN explicit_dimension_type-->

```kotlin
val c: NDArray<Double, D2> = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
```

<!---END-->

## From collections and primitive arrays

Use `mk.ndarray` or helper extensions to convert Kotlin collections and primitive arrays into ndarrays.
This is convenient when you already have data in memory.

<!---FUN from_collections-->

```kotlin
val fromList = mk.ndarray(listOf(10, 20, 30))
// [10, 20, 30]

val fromIterable = listOf(1.2, 3.4, 5.6).toNDArray()
// [1.2, 3.4, 5.6]
```

<!---END-->

If you have a flat primitive array and know the shape, pass it directly:

<!---FUN from_primitive_array-->

```kotlin
val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
val shaped = mk.ndarray(data, 2, 3)
/*
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]]
 */
```

<!---END-->

## Factory functions

Multik provides a set of factory helpers for common patterns:

<!---FUN factory_functions-->

```kotlin
val zeros = mk.zeros<Double>(2, 3)
/*
[[0.0, 0.0, 0.0],
 [0.0, 0.0, 0.0]]
 */

val ones = mk.ones<Int>(3)
// [1, 1, 1]

val eye = mk.identity<Double>(3)
/*
[[1.0, 0.0, 0.0],
 [0.0, 1.0, 0.0],
 [0.0, 0.0, 1.0]]
 */

val diag = mk.diagonal(mk[2, 4, 8])
/*
[[2, 0, 0],
 [0, 4, 0],
 [0, 0, 8]]
 */
```

<!---END-->

For ranges and evenly spaced values:

<!---FUN range_and_linspace-->

```kotlin
val arange = mk.arange<Int>(0, 10, 2)
// [0, 2, 4, 6, 8]

val lin = mk.linspace<Double>(0.0, 1.0, 5)
// [0.0, 0.25, 0.5, 0.75, 1.0]
```

<!---END-->

## Lambda-based creation

Use lambda factories when you want values computed from indices:

<!---FUN lambda_creation-->

```kotlin
val squares = mk.d2array(3, 3) { it * it }
/*
[[0, 1, 4],
 [9, 16, 25],
 [36, 49, 64]]
 */

val grid = mk.d2arrayIndices(2, 3) { i, j -> i * 10 + j }
/*
[[0, 1, 2],
 [10, 11, 12]]
 */
```

<!---END-->

For 3D or higher, use the corresponding helpers (`d3array`, `d4array`, or `ndarray` with a shape).

## Controlling dtype and dimension

Most creation functions infer dtype and dimension, but you can also be explicit for clarity and type-safety.
This helps avoid accidental promotions or dimension mismatches later in your pipeline.

<!---FUN controlling_dtype-->

```kotlin
val x = mk.zeros<Float>(2, 2) // dtype is Float

val y: NDArray<Int, D2> = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
```

<!---END-->

If you need to change element types, see [](type-casting.md).

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="array-fundamentals.md" summary="Understand NDArray fundamentals and basic arithmetic."/>
  <a href="indexing-and-slicing.md" summary="Access elements and subarrays with ranges, slices, and indices."/>
  <a href="shape-manipulation.md" summary="Reshape, transpose, squeeze, and combine arrays."/>
  <a href="copies-and-views.md" summary="Learn when operations return views and how to copy data safely."/>
</category>
</seealso>
