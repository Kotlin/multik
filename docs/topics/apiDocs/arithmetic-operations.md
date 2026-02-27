# Arithmetic operations

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for arithmetic operations on Multik NDArrays — element-wise addition,
subtraction, multiplication, division, and unary negation.
</web-summary>

<card-summary>
Element-wise +, -, *, / and unary minus on NDArrays with scalars or other arrays.
</card-summary>

<link-summary>
Arithmetic operations — element-wise +, -, *, / on NDArrays.
</link-summary>

## Overview

All arithmetic operations are **element-wise** and return a new `NDArray`. Both array-array and
array-scalar forms are supported. In-place variants (`+=`, `-=`, `*=`, `/=`) modify the array directly.

Supported element types: `Byte`, `Short`, `Int`, `Long`, `Float`, `Double`, `ComplexFloat`, `ComplexDouble`.

## Addition

```kotlin
// Array + Array
operator fun <T, D : Dimension> MultiArray<T, D>.plus(other: MultiArray<T, D>): NDArray<T, D>

// Array + Scalar
operator fun <T, D : Dimension> MultiArray<T, D>.plus(other: T): NDArray<T, D>

// Scalar + Array
operator fun <D : Dimension> Int.plus(other: MultiArray<Int, D>): NDArray<Int, D>
// (also for Byte, Short, Long, Float, Double, ComplexFloat, ComplexDouble)

// In-place
operator fun <T, D : Dimension> MutableMultiArray<T, D>.plusAssign(other: MultiArray<T, D>)
operator fun <T, D : Dimension> MutableMultiArray<T, D>.plusAssign(other: T)
```

### Example
{id="Addition-example"}

<!---FUN addition_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
val b = mk.ndarray(mk[10, 20, 30])

val c = a + b        // [11, 22, 33]
val d = a + 100      // [101, 102, 103]
val e = 5 + a        // [6, 7, 8]
```

<!---END-->

## Subtraction

```kotlin
operator fun <T, D : Dimension> MultiArray<T, D>.minus(other: MultiArray<T, D>): NDArray<T, D>
operator fun <T, D : Dimension> MultiArray<T, D>.minus(other: T): NDArray<T, D>
operator fun <D : Dimension> Int.minus(other: MultiArray<Int, D>): NDArray<Int, D>

// In-place
operator fun <T, D : Dimension> MutableMultiArray<T, D>.minusAssign(other: MultiArray<T, D>)
operator fun <T, D : Dimension> MutableMultiArray<T, D>.minusAssign(other: T)
```

### Example
{id="Subtraction-example"}

<!---FUN subtraction_example-->

```kotlin
val a = mk.ndarray(mk[10, 20, 30])
val b = mk.ndarray(mk[1, 2, 3])

val c = a - b        // [9, 18, 27]
val d = a - 5        // [5, 15, 25]
val e = 100 - a      // [90, 80, 70]
```

<!---END-->

## Multiplication

```kotlin
operator fun <T, D : Dimension> MultiArray<T, D>.times(other: MultiArray<T, D>): NDArray<T, D>
operator fun <T, D : Dimension> MultiArray<T, D>.times(other: T): NDArray<T, D>
operator fun <D : Dimension> Int.times(other: MultiArray<Int, D>): NDArray<Int, D>

// In-place
operator fun <T, D : Dimension> MutableMultiArray<T, D>.timesAssign(other: MultiArray<T, D>)
operator fun <T, D : Dimension> MutableMultiArray<T, D>.timesAssign(other: T)
```

> This is **element-wise** multiplication, not matrix multiplication. For matrix multiplication, use
> [`mk.linalg.dot()`](dot.md).
> {style="warning"}

### Example
{id="Multiplication-example"}

<!---FUN multiplication_example-->

```kotlin
val a = mk.ndarray(mk[2, 3, 4])
val b = mk.ndarray(mk[10, 20, 30])

val c = a * b        // [20, 60, 120]
val d = a * 3        // [6, 9, 12]
```

<!---END-->

## Division

```kotlin
operator fun <T, D : Dimension> MultiArray<T, D>.div(other: MultiArray<T, D>): NDArray<T, D>
operator fun <T, D : Dimension> MultiArray<T, D>.div(other: T): NDArray<T, D>
operator fun <D : Dimension> Int.div(other: MultiArray<Int, D>): NDArray<Int, D>

// In-place
operator fun <T, D : Dimension> MutableMultiArray<T, D>.divAssign(other: MultiArray<T, D>)
operator fun <T, D : Dimension> MutableMultiArray<T, D>.divAssign(other: T)
```

### Example
{id="Division-example"}

<!---FUN division_example-->

```kotlin
val a = mk.ndarray(mk[10.0, 20.0, 30.0])

val b = a / 2.0      // [5.0, 10.0, 15.0]
```

<!---END-->

## Unary negation

```kotlin
operator fun <T, D : Dimension> MultiArray<T, D>.unaryMinus(): NDArray<T, D>
```

### Example

<!---FUN unary_negation_example-->

```kotlin
val a = mk.ndarray(mk[1, -2, 3])
val b = -a           // [-1, 2, -3]
```

<!---END-->

## In-place operations

Use the assign variants to modify an array without creating a copy:

<!---FUN in_place_arithmetic_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
a += 10              // a is now [11, 12, 13]
a *= 2               // a is now [22, 24, 26]
```

<!---END-->

## Pitfalls

* Both operands in array-array operations must have the **same shape**. Broadcasting is not supported.
* Integer division truncates — `mk.ndarray(mk[5]) / 2` yields `[2]`, not `[2.5]`.

<seealso style="cards">
<category ref="api-docs">
  <a href="logical-operations.md" summary="Element-wise and, or operations."/>
  <a href="comparison-operations.md" summary="Element-wise minimum and maximum."/>
  <a href="dot.md" summary="Matrix multiplication with mk.linalg.dot()."/>
</category>
</seealso>
