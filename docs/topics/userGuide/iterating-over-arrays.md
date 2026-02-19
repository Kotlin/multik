# Iterating over arrays

<!---IMPORT samples.docs.userGuide.IteratingOverArrays-->

<web-summary>
Learn how to iterate over ndarrays in Multik, from simple element loops to multi-index traversal.
Use built-in helpers like `forEach`, `forEachIndexed`, and `forEachMultiIndexed` to keep code clean.
</web-summary>

<card-summary>
Iterate through ndarrays element-wise or with full indices.
Use helper functions for concise and safe traversal.
</card-summary>

<link-summary>
Iterate over ndarrays with element loops, indices, and multi-index helpers.
</link-summary>

## Overview

Iteration in Multik is element-wise by default.
Use indices or multi-indices when you need coordinates along each axis.

## Element-wise iteration

You can iterate an ndarray just like a Kotlin collection:

<!---FUN elementWise_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
for (value in a) {
    print("$value ")
}
// 1 2 3
```

<!---END-->

Multik also provides `forEach` for concise iteration:

<!---FUN forEach_example-->

```kotlin
a.forEach { value ->
    println(value)
}
```

<!---END-->


Iteration walks the array in logical index order (the last axis changes fastest) and respects views/strides.

## Iterating with indices (1D)

For one-dimensional arrays, use `indices` or `forEachIndexed`:

<!---FUN indices_example-->

```kotlin
val v = mk.ndarray(mk[10, 20, 30])

for (i in v.indices) {
    println("v[$i] = ${v[i]}")
}

v.forEachIndexed { i, value ->
    println("$i -> $value")
}
```

<!---END-->

## Iterating with multi-indices (ND)

For multidimensional arrays, `multiIndices` gives all index tuples.
Each index is an `IntArray` that you can pass back into the array.

<!---FUN multiIndices_example-->

```kotlin
val m = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])

for (index in m.multiIndices) {
    print("${m[index]} ")
}
// 1.5 2.1 3.0 4.0 5.0 6.0
```

<!---END-->

There is also a helper that combines iteration with indices:

<!---FUN forEachMultiIndexed_example-->

```kotlin
m.forEachMultiIndexed { index, value ->
    println("${index.joinToString()} -> $value")
}
```

<!---END-->

## Updating values in-place

`NDArray` is mutable, so you can update elements as you iterate:

<!---FUN update_example-->

```kotlin
val x = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
for (index in x.multiIndices) {
    x[index] = x[index] * 2
}
// [[2, 4], [6, 8]]
```

<!---END-->

If you need an independent copy before modifying, use `copy()` or `deepCopy()`.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="indexing-and-slicing.md" summary="Access elements and subarrays with ranges, slices, and indices."/>
  <a href="copies-and-views.md" summary="Learn when operations return views and how to copy safely."/>
  <a href="performance-and-optimization.md" summary="Reduce allocations and work efficiently with ndarrays."/>
</category>
</seealso>
