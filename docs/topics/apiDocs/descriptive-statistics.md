# Descriptive statistics

<web-summary>
API reference for descriptive statistics in Multik — mean, median, and weighted average
via mk.stat.
</web-summary>

<card-summary>
Descriptive statistics — mean, median, and weighted average.
</card-summary>

<link-summary>
Descriptive statistics — mean, median, average.
</link-summary>

## Overview

Multik provides three core descriptive statistics functions through `mk.stat`:

| Function  | Description                                             |
|-----------|---------------------------------------------------------|
| `mean`    | Arithmetic mean of all elements, or along a given axis. |
| `median`  | Median (middle value) of all elements.                  |
| `average` | Weighted average over all elements.                     |

## mean

{id="mean"}

Returns the arithmetic mean of all elements in the array, or along a given axis.

### Signatures

{id="mean-signatures"}

```kotlin
// Scalar mean (all elements)
fun <T : Number, D : Dimension> Statistics.mean(a: MultiArray<T, D>): Double

// Mean along axis
fun <T : Number, D : Dimension, O : Dimension> Statistics.mean(
    a: MultiArray<T, D>,
    axis: Int
): NDArray<Double, O>
```

Dimension-specific overloads are also available:

```kotlin
fun <T : Number> Statistics.meanD2(a: MultiArray<T, D2>, axis: Int): NDArray<Double, D1>
fun <T : Number> Statistics.meanD3(a: MultiArray<T, D3>, axis: Int): NDArray<Double, D2>
fun <T : Number> Statistics.meanD4(a: MultiArray<T, D4>, axis: Int): NDArray<Double, D3>
fun <T : Number> Statistics.meanDN(a: MultiArray<T, DN>, axis: Int): NDArray<Double, DN>
```

### Parameters

{id="mean-params"}

| Parameter | Type               | Description                       |
|-----------|--------------------|-----------------------------------|
| `a`       | `MultiArray<T, D>` | Input array.                      |
| `axis`    | `Int`              | Axis along which to compute mean. |

**Returns:** `Double` for the scalar form, or `NDArray<Double, O>` for the axis form.

### Example

{id="mean-example"}

```kotlin
val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

mk.stat.mean(a)                                  // 3.5
mk.stat.meanD2(a, axis = 0)                      // [2.5, 3.5, 4.5]
mk.stat.meanD2(a, axis = 1)                      // [2.0, 5.0]
```

## median

{id="median"}

Returns the median of all elements in the array. Elements are sorted and the middle value is
returned (or the average of the two middle values for even-length arrays).

### Signature

{id="median-signature"}

```kotlin
fun <T : Number, D : Dimension> Statistics.median(a: MultiArray<T, D>): Double?
```

### Parameters

{id="median-params"}

| Parameter | Type               | Description  |
|-----------|--------------------|--------------|
| `a`       | `MultiArray<T, D>` | Input array. |

**Returns:** `Double?` — the median value, or `null` if the array is empty.

### Example

{id="median-example"}

```kotlin
val a = mk.ndarray(mk[3, 1, 4, 1, 5, 9, 2, 6])

mk.stat.median(a)   // 3.5  (average of 3 and 4 after sorting)

val b = mk.ndarray(mk[1, 2, 3])
mk.stat.median(b)   // 2.0
```

## average

{id="average"}

Returns the weighted average of all elements. When no weights are provided, all elements are
weighted equally (equivalent to `mean`).

### Signature

{id="average-signature"}

```kotlin
fun <T : Number, D : Dimension> Statistics.average(
    a: MultiArray<T, D>,
    weights: MultiArray<T, D>? = null
): Double
```

### Parameters

{id="average-params"}

| Parameter | Type                | Description                                                  |
|-----------|---------------------|--------------------------------------------------------------|
| `a`       | `MultiArray<T, D>`  | Input array.                                                 |
| `weights` | `MultiArray<T, D>?` | Optional weights array (same shape as `a`). Default: `null`. |

**Returns:** `Double` — the weighted average.

### Example

{id="average-example"}

```kotlin
val values = mk.ndarray(mk[1.0, 2.0, 3.0, 4.0])

// Unweighted (equivalent to mean)
mk.stat.average(values)                                      // 2.5

// Weighted
val weights = mk.ndarray(mk[1.0, 1.0, 1.0, 5.0])
mk.stat.average(values, weights)                             // 3.25
// Calculation: (1*1 + 2*1 + 3*1 + 4*5) / (1+1+1+5) = 26/8 = 3.25
```

## Pitfalls

* `mean` always returns `Double`, even for integer arrays.
* `median` returns `Double?` — it can be `null` for empty arrays.
* Axis-based `mean` overloads require explicit dimension type parameters, e.g. `mk.stat.meanD2(a, axis = 0)`.
* The `weights` array in `average` must have the **same shape** as the input array.

<seealso style="cards">
<category ref="api-docs">
  <a href="statistics.md" summary="Overview of all statistics functions."/>
  <a href="mathematical.md" summary="Mathematical functions via mk.math."/>
  <a href="arithmetic-operations.md" summary="Element-wise arithmetic operations."/>
</category>
</seealso>