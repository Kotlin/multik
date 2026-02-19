# Copies and views

<!---IMPORT samples.docs.userGuide.CopiesAndViews-->

<web-summary>
Understand when Multik returns views that share data and when it creates new arrays.
Learn how to use copy and deepCopy to control memory ownership and layout.
</web-summary>

<card-summary>
Views share data; copies own it.
Learn which operations allocate new storage and how to detach safely.
</card-summary>

<link-summary>
Learn the difference between views and copies, and how to control ownership with copy APIs.
</link-summary>

## Overview

A **view** is an ndarray that references the same underlying data as another array.
A **copy** owns its own storage and is independent from the original.
Views are fast and memory-efficient, but changes to the base array are visible through the view.

## Creating views

Most indexing and slicing operations return views:

<!---FUN creating_views-->

```kotlin
val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
val v = a[0] // view of the first row

println(v) // [1, 2]

a[0, 0] = 99
println(v) // [99, 2]
```

<!---END-->

You can also create views explicitly with `view` and `slice`.
Use `base` to check whether an array is a view:

<!---FUN view_with_slice-->

```kotlin
val s = a.slice<Int, D2, D2>(0..1, axis = 1)
println(s.base != null) // true for views
```

<!---END-->

## Copies: `copy` and `deepCopy`

Use these methods to detach from a view and get independent data:

* `copy()` duplicates the underlying storage and preserves the current layout (offset/strides).
* `deepCopy()` materializes only the visible elements into a new contiguous array.

<!---FUN copy_and_deepCopy-->

```kotlin
val view = a[0]
val c1 = view.copy()
val c2 = view.deepCopy()

a[0, 0] = 111
println(view) // changed: [111, 2]
println(c1)   // unchanged: [1, 2]
println(c2)   // unchanged: [1, 2]
```

<!---END-->

`deepCopy()` is usually the safer choice when you need a compact, standalone array
or want to avoid copying data that is outside a view.

## Views vs copies in common operations

Typical view-producing operations:

* Indexing and slicing
* `view(...)` and `slice(...)`
* `transpose()` and `squeeze()`

Operations that always allocate new storage:

* `copy()` and `deepCopy()`
* `flatten()`
* `cat(...)` and `mk.stack(...)`

Operations like `reshape()` and `unsqueeze()` return views when the array is contiguous;
otherwise they materialize a copy to keep data consistent.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="indexing-and-slicing.md" summary="Access elements and subarrays with ranges, slices, and indices."/>
  <a href="shape-manipulation.md" summary="Reshape, transpose, squeeze, and combine arrays."/>
  <a href="performance-and-optimization.md" summary="Reduce allocations and work efficiently with ndarrays."/>
</category>
</seealso>