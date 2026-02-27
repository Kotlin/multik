# Comparison operations

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for comparison operations on Multik NDArrays — element-wise minimum
and maximum between two arrays.
</web-summary>

<card-summary>
Element-wise minimum and maximum comparison between two NDArrays.
</card-summary>

<link-summary>
Comparison operations — element-wise minimum and maximum.
</link-summary>

## Overview

Comparison operations take two arrays of the same shape and return a new array where each element
is the result of an element-wise comparison. Multik provides `minimum` and `maximum`.

## minimum

```kotlin
fun <T : Number, D : Dimension> MultiArray<T, D>.minimum(
    other: MultiArray<T, D>
): NDArray<T, D>
```

Returns a new array where each element is the smaller of the two corresponding elements.

### Example
{id="minimum-example"}

<!---FUN minimum_comparison_example-->

```kotlin
val a = mk.ndarray(mk[1, 5, 3])
val b = mk.ndarray(mk[4, 2, 6])

val c = a.minimum(b)  // [1, 2, 3]
```

<!---END-->

## maximum

```kotlin
fun <T : Number, D : Dimension> MultiArray<T, D>.maximum(
    other: MultiArray<T, D>
): NDArray<T, D>
```

Returns a new array where each element is the larger of the two corresponding elements.

### Example
{id="maximum-example"}

<!---FUN maximum_comparison_example-->

```kotlin
val a = mk.ndarray(mk[1, 5, 3])
val b = mk.ndarray(mk[4, 2, 6])

val c = a.maximum(b)  // [4, 5, 6]
```

<!---END-->

## Pitfalls

* Both arrays must have the **same shape**.
* Complex types (`ComplexFloat`, `ComplexDouble`) are **not** supported — throws `UnsupportedOperationException`.
* For single-array min/max, see [`max`](max.md) and [`min`](min.md).

<seealso style="cards">
<category ref="api-docs">
  <a href="max.md" summary="Find the largest element in an array."/>
  <a href="min.md" summary="Find the smallest element in an array."/>
  <a href="arithmetic-operations.md" summary="Element-wise +, -, *, / operations."/>
  <a href="logical-operations.md" summary="Element-wise and, or operations."/>
</category>
</seealso>
