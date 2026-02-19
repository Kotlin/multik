# Shape manipulation

<!---IMPORT samples.docs.userGuide.ShapeManipulation-->

<web-summary>
Learn how to change ndarray shapes in Multik: reshape, flatten, transpose, and axis operations.
Understand which operations return views and which create new arrays.
</web-summary>

<card-summary>
Reshape, transpose, squeeze, and combine ndarrays.
See how shapes change and when data is copied.
</card-summary>

<link-summary>
Manipulate ndarray shapes with reshape, transpose, squeeze, and concatenation.
</link-summary>

## Overview

Shape manipulation changes how the same data is interpreted across dimensions.
Some operations return views over the same storage, while others create new arrays.
Use `arr.shape` and `arr.dim` to verify the resulting layout.

## Reshape

`reshape` reinterprets the data with a new shape without changing element order.
The new shape must contain the same number of elements as the original array.

<!---FUN reshape_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6])
val b = a.reshape(2, 3)
/*
[[1, 2, 3],
 [4, 5, 6]]
 */

val c = b.reshape(3, 2)
/*
[[1, 2],
 [3, 4],
 [5, 6]]
 */
```

<!---END-->

There is no `-1` dimension inference, so all dimensions must be specified explicitly.
When possible, `reshape` returns a view; otherwise it may copy data to keep storage consistent.

## Flatten

`flatten` always returns a one-dimensional copy of the array.

<!---FUN flatten_example-->

```kotlin
val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
val flat = m.flatten()
// [1, 2, 3, 4]
```

<!---END-->

## Transpose

`transpose` reorders axes. With no arguments, it reverses the axes order.

<!---FUN transpose_example-->

```kotlin
val m = mk.ndarray(mk[mk[1, 2], mk[3, 4], mk[5, 6]])
val t = m.transpose()
/*
[[1, 3, 5],
 [2, 4, 6]]
 */
```

<!---END-->

For higher dimensions, provide the new axes order:

<!---FUN transpose_2_example-->

```kotlin
val x = mk.d3array(2, 3, 4) { it }
val y = x.transpose(1, 0, 2) // swap axes 0 and 1
```

<!---END-->

`transpose` returns a view with new shape and strides.

## Squeeze and unsqueeze

`squeeze` removes axes of size 1. Use it to drop redundant dimensions.

<!---FUN squeeze_example-->

```kotlin
val a = mk.zeros<Double>(1, 3, 1)
val b = a.squeeze()
// shape is (3)
```

<!---END-->

`unsqueeze` inserts axes of size 1 at the specified positions:

<!---FUN unsqueeze_example-->

```kotlin
val v = mk.ndarray(mk[1, 2, 3])
val col = v.unsqueeze(1) // shape (3, 1)
val row = v.unsqueeze(0) // shape (1, 3)
```

<!---END-->

You can also use `expandDims` / `expandNDims` as helpers for adding size-1 axes.

## Concatenate and stack

Use `cat` to concatenate arrays along an axis. The shapes must match on all other axes.

<!---FUN cat_example-->

```kotlin
val left = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
val right = mk.ndarray(mk[mk[5, 6], mk[7, 8]])

val byRows = left.cat(right, axis = 0)
/*
[[1, 2],
 [3, 4],
 [5, 6],
 [7, 8]]
 */

val byCols = left.cat(right, axis = 1)
/*
[[1, 2, 5, 6],
 [3, 4, 7, 8]]
 */
```

<!---END-->

Use `mk.stack` to add a new axis and combine arrays of the same shape:

<!---FUN stack_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
val b = mk.ndarray(mk[4, 5, 6])
val stacked = mk.stack(a, b) // shape (2, 3)
```

<!---END-->

`cat` and `stack` create new arrays and copy data.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="indexing-and-slicing.md" summary="Access elements and subarrays with ranges, slices, and indices."/>
  <a href="copies-and-views.md" summary="Learn when operations return views and how to copy safely."/>
  <a href="iterating-over-arrays.md" summary="Iterate element-wise and with multi-indices."/>
</category>
</seealso>
