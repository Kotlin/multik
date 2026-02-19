# Mathematical

<web-summary>
API reference for mathematical functions in Multik — aggregation (argMax, argMin, max, min, sum, cumSum),
transcendental (sin, cos, exp, log), and absolute value.
</web-summary>

<card-summary>
Aggregation, transcendental, and absolute-value functions via mk.math.
</card-summary>

<link-summary>
Mathematical functions — aggregation, transcendental, and abs.
</link-summary>

## Overview

The `mk.math` entry point provides element-wise mathematical operations and array aggregation functions.
Functions are defined in the `Math` and `MathEx` interfaces; convenience extension functions on `MultiArray`
are also available.

Supported element types: `Byte`, `Short`, `Int`, `Long`, `Float`, `Double`, `ComplexFloat`, `ComplexDouble`
(transcendental functions only support `Number`, `Float`, `ComplexFloat`, `ComplexDouble`).

## Aggregation functions

| Function | Description                                                  |
|----------|--------------------------------------------------------------|
| `argMax` | Flat index of the maximum element, or indices along an axis. |
| `argMin` | Flat index of the minimum element, or indices along an axis. |
| `max`    | Maximum element, or max along an axis.                       |
| `min`    | Minimum element, or min along an axis.                       |
| `sum`    | Sum of all elements, or sum along an axis.                   |
| `cumSum` | Cumulative sum (flat or along an axis).                      |

### Signatures

{id="aggregation-signatures"}

```kotlin
// argMax / argMin — flat index
fun <T : Number, D : Dimension> Math.argMax(a: MultiArray<T, D>): Int
fun <T : Number, D : Dimension> Math.argMin(a: MultiArray<T, D>): Int

// argMax / argMin — along axis
fun <T : Number, D : Dimension, O : Dimension> Math.argMax(a: MultiArray<T, D>, axis: Int): NDArray<Int, O>
fun <T : Number, D : Dimension, O : Dimension> Math.argMin(a: MultiArray<T, D>, axis: Int): NDArray<Int, O>

// max / min
fun <T : Number, D : Dimension> Math.max(a: MultiArray<T, D>): T
fun <T : Number, D : Dimension, O : Dimension> Math.max(a: MultiArray<T, D>, axis: Int): NDArray<T, O>
fun <T : Number, D : Dimension> Math.min(a: MultiArray<T, D>): T
fun <T : Number, D : Dimension, O : Dimension> Math.min(a: MultiArray<T, D>, axis: Int): NDArray<T, O>

// sum
fun <T : Number, D : Dimension> Math.sum(a: MultiArray<T, D>): T
fun <T : Number, D : Dimension, O : Dimension> Math.sum(a: MultiArray<T, D>, axis: Int): NDArray<T, O>

// cumSum
fun <T : Number, D : Dimension> Math.cumSum(a: MultiArray<T, D>): D1Array<T>
fun <T : Number, D : Dimension> Math.cumSum(a: MultiArray<T, D>, axis: Int): NDArray<T, D>
```

### Example

{id="aggregation-example"}

```kotlin
val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

mk.math.argMax(a)           // 5  (flat index of element 6)
mk.math.max(a)              // 6
mk.math.sum(a)              // 21
mk.math.cumSum(a)           // [1, 3, 6, 10, 15, 21]

// Along axis
mk.math.sum<Int, D2, D1>(a, axis = 0)   // [5, 7, 9]
mk.math.sum<Int, D2, D1>(a, axis = 1)   // [6, 15]
```

### Extension form

{id="aggregation-extension"}

`argMax`, `argMin`, and `cumSum` are also available as extension functions on `MultiArray`:

```kotlin
val a = mk.ndarray(mk[3, 1, 4, 1, 5])

a.argMax()   // 4
a.argMin()   // 1
a.cumSum()   // [3, 4, 8, 9, 14]
```

## Transcendental functions

| Function | Description                          |
|----------|--------------------------------------|
| `sin`    | Element-wise sine.                   |
| `cos`    | Element-wise cosine.                 |
| `exp`    | Element-wise exponential (e^x).      |
| `log`    | Element-wise natural logarithm (ln). |

### Signatures

{id="transcendental-signatures"}

```kotlin
// Default — returns Double
fun <T : Number, D : Dimension> MathEx.sin(a: MultiArray<T, D>): NDArray<Double, D>
fun <T : Number, D : Dimension> MathEx.cos(a: MultiArray<T, D>): NDArray<Double, D>
fun <T : Number, D : Dimension> MathEx.exp(a: MultiArray<T, D>): NDArray<Double, D>
fun <T : Number, D : Dimension> MathEx.log(a: MultiArray<T, D>): NDArray<Double, D>

// Float overloads
fun <D : Dimension> MathEx.sinF(a: MultiArray<Float, D>): NDArray<Float, D>
fun <D : Dimension> MathEx.cosF(a: MultiArray<Float, D>): NDArray<Float, D>
fun <D : Dimension> MathEx.expF(a: MultiArray<Float, D>): NDArray<Float, D>
fun <D : Dimension> MathEx.logF(a: MultiArray<Float, D>): NDArray<Float, D>

// ComplexFloat / ComplexDouble overloads
fun <D : Dimension> MathEx.sinCF(a: MultiArray<ComplexFloat, D>): NDArray<ComplexFloat, D>
fun <D : Dimension> MathEx.sinCD(a: MultiArray<ComplexDouble, D>): NDArray<ComplexDouble, D>
// (same pattern for cos, exp, log)
```

### Example

{id="transcendental-example"}

```kotlin
val a = mk.ndarray(mk[0.0, Math.PI / 2, Math.PI])

mk.math.mathEx.sin(a)   // [0.0, 1.0, ~0.0]
mk.math.mathEx.cos(a)   // [1.0, ~0.0, -1.0]
mk.math.mathEx.exp(a)   // [1.0, 4.8104..., 23.1406...]
mk.math.mathEx.log(mk.ndarray(mk[1.0, Math.E, 10.0]))  // [0.0, 1.0, 2.302...]
```

### Extension form

{id="transcendental-extension"}

Transcendental functions are also available as extensions on `MultiArray`:

```kotlin
val a = mk.ndarray(mk[1.0, 2.0, 3.0])

a.sin()   // element-wise sine → NDArray<Double, D1>
a.cos()   // element-wise cosine
a.exp()   // element-wise exp
a.log()   // element-wise natural log
```

## Absolute value

The `abs` function returns an ndarray with the absolute value of each element. It is a standalone function
(not part of `mk.math`).

### Signatures

{id="abs-signatures"}

```kotlin
fun <D : Dimension> abs(a: MultiArray<Byte, D>): NDArray<Byte, D>
fun <D : Dimension> abs(a: MultiArray<Short, D>): NDArray<Short, D>
fun <D : Dimension> abs(a: MultiArray<Int, D>): NDArray<Int, D>
fun <D : Dimension> abs(a: MultiArray<Long, D>): NDArray<Long, D>
fun <D : Dimension> abs(a: MultiArray<Float, D>): NDArray<Float, D>
fun <D : Dimension> abs(a: MultiArray<Double, D>): NDArray<Double, D>
fun <D : Dimension> abs(a: MultiArray<ComplexFloat, D>): NDArray<Float, D>
fun <D : Dimension> abs(a: MultiArray<ComplexDouble, D>): NDArray<Double, D>
```

> For complex types, `abs` returns the magnitude as a real-valued array (`Float` or `Double`).
> {style="note"}

### Example

{id="abs-example"}

```kotlin
val a = mk.ndarray(mk[-3, -1, 0, 2, -5])
abs(a)   // [3, 1, 0, 2, 5]
```

## Pitfalls

* Axis-based overloads (`argMax`, `max`, `min`, `sum` with `axis`) require explicit type parameters for the output
  dimension, e.g. `mk.math.sum<Int, D2, D1>(a, axis = 0)`.
* `cumSum()` without an axis always returns a flat `D1Array`, regardless of input dimensionality.

<seealso style="cards">
<category ref="api-docs">
  <a href="arithmetic-operations.md" summary="Element-wise +, -, *, / on NDArrays."/>
  <a href="linear-algebra.md" summary="Linear algebra operations via mk.linalg."/>
  <a href="statistics.md" summary="Statistical functions via mk.stat."/>
</category>
</seealso>