# Indexing and slicing

<!---IMPORT samples.docs.userGuide.IndexingAndSlicing-->

<web-summary>
Learn how to access elements and subarrays in Multik using indexing and slicing.
Understand ranges, steps, open-ended slices, and how dimensions change when you index or slice.
</web-summary>

<card-summary>
Index and slice ndarrays with Kotlin-friendly syntax.
Use ranges, steps, and open-ended slices while controlling dimensionality.
</card-summary>

<link-summary>
Access elements and subarrays with Multik indexing and slicing.
</link-summary>

## Overview

Indexing selects elements or subarrays, while slicing selects ranges along one or more axes.
This section covers ranges, steps, open-ended slices, and how dimensionality changes.

## Indexing basics

Multik uses zero-based indexing. Each index corresponds to an axis of the ndarray.

<!---FUN indexing_basics-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
a[2] // 3

val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
b[1, 2] // 6.0
```

<!---END-->

When you index a single axis with an `Int`, the resulting array has one less dimension.

<!---FUN dimension_reduction-->

```kotlin
val row = b[1] // D1 array: [4.0, 5.0, 6.0]
```

<!---END-->

## Slicing with ranges

Use Kotlin ranges to take slices. Ranges are inclusive, and you can use `..<` (or `until`) for exclusive end bounds.

<!---FUN slicing_with_ranges-->

```kotlin
val c = mk.ndarray(mk[10, 20, 30, 40, 50])

c[1..3]   // [20, 30, 40]
c[0..<3]  // [10, 20, 30]
```

<!---END-->

You can also specify a step using the `..` operator again:

<!---FUN step_slicing-->

```kotlin
c[0..4..2] // [10, 30, 50]
```

<!---END-->

## Slicing multiple axes

For multidimensional arrays, pass a range per axis. If you omit axes, Multik treats them as full slices.

<!---FUN multi_axis_slicing-->

```kotlin
// take rows 0 and 1 from column 1
b[0..<2, 1] // [2.1, 5.0]

// same as selecting the entire second axis
b[1]         // [4.0, 5.0, 6.0]
b[1, 0..2..1] // [4.0, 5.0, 6.0]
```

<!---END-->

## Open-ended slices with `sl`

For open-ended slices, use `sl` helpers:

* `sl.first..k` - from the start to `k`
* `k..sl.last` - from `k` to the end
* `sl.bounds` - the whole axis

<!---FUN open_ended_slices-->

```kotlin
val col = b[sl.bounds, 1] // full rows, column 1
val head = c[sl.first..2] // [10, 20, 30]
val tail = c[2..sl.last]  // [30, 40, 50]
```

<!---END-->

## Keeping or reducing dimensions

Indexing with an `Int` removes the corresponding axis.
If you want to keep the axis, slice a range of length 1 instead.

<!---FUN keeping_dimensions-->

```kotlin
val keepDim = b[1..1] // D2 array with shape (1, 3)
```

<!---END-->

## Slicing by axis and ND arrays

When you need to slice an arbitrary axis or a general ND array, use `slice`:

<!---FUN slicing_by_axis-->

```kotlin
val d = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

val byAxis = d.slice<Int, D2, D2>(0..1, axis = 1) // take columns 0..1
```

<!---END-->

For ND arrays, you can provide a map of axis to slice or index:

<!---FUN nd_slicing-->

```kotlin
val e = mk.d3array(2, 3, 4) { it }
val view = e.slice<Int, D3, D2>(mapOf(0 to 0..1.r, 2 to (0..2).toSlice()))
// axis 0 is indexed, axis 2 is sliced, axis 1 is kept as full slice
```

<!---END-->

## Views vs copies

Slicing returns a view backed by the original data.
If you need an independent copy, see [](copies-and-views.md).

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="iterating-over-arrays.md" summary="Iterate element-wise and with multi-indices."/>
  <a href="copies-and-views.md" summary="Learn when slicing returns views and how to copy safely."/>
</category>
<category ref="api-docs">
  <a href="indexing-routines.md" summary="Indexing helpers and convenience APIs.">Indexing Routines</a>
</category>
</seealso>
