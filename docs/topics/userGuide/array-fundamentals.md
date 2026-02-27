# Array fundamentals

<!---IMPORT samples.docs.userGuide.Basic-->

<web-summary>
Learn the core ideas of Multik: what an NDArray is, how dimensions and shapes work,
and how to perform basic element-wise operations in Kotlin.
</web-summary>

<card-summary>
Understand NDArray fundamentals, key properties, and the basic arithmetic workflow.
</card-summary>

<link-summary>
Get the essentials of NDArray, shapes, dtypes, and simple operations in Multik.
</link-summary>

## Overview

This page introduces the NDArray model, its core properties, and the basic arithmetic
and indexing workflow you will use throughout Multik.

## NDArray essentials

An `NDArray` is Multik's core data type for dense, homogeneous numeric data.
It is defined by four key properties:

* **Dimension** (`dim`) — number of axes, such as `D1`, `D2`, `D3`, `D4`, or `DN`.
* **Shape** (`shape`) — sizes of each axis, for example `(2, 3)`.
* **Size** (`size`) — total number of elements.
* **DType** (`dtype`) — element type, such as `Int`, `Double`, or complex types.

These properties help keep computations correct and predictable.

## Creating arrays

For quick creation, use `mk[]` with `mk.ndarray` and let Multik infer type and dimension:

<!---FUN creating_arrays-->

```kotlin
val v = mk.ndarray(mk[1, 2, 3])
val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
```

<!---END-->

You can also use factory methods like `zeros`, `ones`, and `identity`.
For a deeper overview, see [](creating-multidimensional-arrays.md).

## Inspecting properties

<!---FUN inspecting_properties-->

```kotlin
val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])

m.shape // (2, 2)
m.size  // 4
m.dim   // D2
m.dtype // DataType.DoubleDataType
```

<!---END-->

## Element-wise arithmetic

Arithmetic is element-wise by default:

<!---FUN elementWise_arithmetic-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
val b = mk.ndarray(mk[4, 5, 6])

val sum = a + b   // [5, 7, 9]
val scaled = a * 2 // [2, 4, 6]
```

<!---END-->

When operating on two arrays, their shapes must match.
For fixed dimensions (D1-D4), dimension mismatch is caught at compile time;
shape mismatch is checked at runtime.

### Matrix product with `dot`

`*` is element-wise. For matrix multiplication use `dot`:

```kotlin
val x = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
val y = mk.ndarray(mk[mk[5.0, 6.0], mk[7.0, 8.0]])

val prod = x dot y
```

## In-place operations

Operations like `+=`, `-=`, `*=`, and `/=` modify the current array:

<!---FUN inPlace_operations-->

```kotlin
val x = mk.ndarray(mk[1, 2, 3])

x += 10
// [11, 12, 13]
```

<!---END-->

## Indexing and slicing

Indexing uses zero-based indices and works per axis:

<!---FUN indexing_basics-->

```kotlin
val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])

m[1, 0] // 3.0
```

<!---END-->

For ranges, slices, and multi-axis selection, see [](indexing-and-slicing.md).

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="creating-multidimensional-arrays.md" summary="Build ndarrays from literals, collections, primitives, and factories."/>
  <a href="indexing-and-slicing.md" summary="Access elements and subarrays with ranges, slices, and indices."/>
  <a href="shape-manipulation.md" summary="Reshape, transpose, squeeze, and combine arrays."/>
  <a href="type-casting.md" summary="Convert ndarrays between numeric and complex dtypes."/>
</category>
</seealso>
