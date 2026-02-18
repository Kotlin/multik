# Indexing routines

<!---IMPORT samples.docs.apiDocs.ArrayObjects-->

<web-summary>
API reference for Multik indexing and slicing: Slice, RInt, ReadableView,
the view and slice extension functions, and open-ended slice helpers.
</web-summary>

<card-summary>
Slice, RInt, ReadableView, and the view/slice extension functions for NDArrays.
</card-summary>

<link-summary>
Indexing API — Slice, RInt, views, and slice helpers.
</link-summary>

## Slice

A `Slice` represents a start–stop–step range along one axis.

```kotlin
public class Slice(start: Int, stop: Int, step: Int) : Indexing, ClosedRange<Int>
```

| Property | Type  | Description                                               |
|----------|-------|-----------------------------------------------------------|
| `start`  | `Int` | First index (inclusive). `-1` means "from the beginning". |
| `stop`   | `Int` | Last index (inclusive). `-1` means "to the end".          |
| `step`   | `Int` | Stride between selected elements. Must be positive.       |

### Creating slices

| Expression         | Result             | Description      |
|--------------------|--------------------|------------------|
| `0.r..4.r`         | `Slice(0, 4, 1)`   | Range via `RInt` |
| `(0..4).toSlice()` | `Slice(0, 4, 1)`   | From `IntRange`  |
| `0..4..2`          | `Slice(0, 4, 2)`   | Range with step  |
| `sl.first..3`      | `Slice(-1, 3, 1)`  | From start to 3  |
| `2..sl.last`       | `Slice(2, -1, 1)`  | From 2 to end    |
| `sl.bounds`        | `Slice(-1, -1, 1)` | Entire axis      |

## RInt

`RInt` (Rankable Int) is a value class that enables the `..` operator to produce `Slice` objects instead of `IntRange`.

```kotlin
public value class RInt(internal val data: Int) : Indexing
```

Convert an `Int` to `RInt` with the `.r` extension:

<!---FUN slice_rInt_example-->

```kotlin
val s = 0.r..4.r          // Slice(0, 4, 1)
val s2 = 0.r until 4.r    // Slice(0, 3, 1)
```

<!---END-->

`RInt` supports `+`, `-`, `*`, `/`, `..`, and `until`.

## sl companion

The `sl` typealias provides open-ended slice boundaries:

| Member      | Type             | Usage                          |
|-------------|------------------|--------------------------------|
| `sl.first`  | `SliceStartStub` | Unbounded start: `sl.first..k` |
| `sl.last`   | `SliceEndStub`   | Unbounded end: `k..sl.last`    |
| `sl.bounds` | `Slice`          | Full axis: `sl.bounds`         |

<!---FUN slice_sl_example-->

```kotlin
val a = mk.ndarray(mk[10, 20, 30, 40, 50])
a[sl.first..2]  // [10, 20, 30]
a[2..sl.last]   // [30, 40, 50]
a[sl.bounds]    // [10, 20, 30, 40, 50]
```

<!---END-->

## ReadableView

`ReadableView` wraps a `DN` array and provides multi-index access through the `V` property.

```kotlin
public class ReadableView<T>(private val base: MultiArray<T, DN>)
```

```kotlin
operator fun get(vararg indices: Int): MultiArray<T, DN>
```

Access via the `V` property:

<!---FUN view_v_property_example-->

```kotlin
val tensor = mk.d3array(2, 3, 4) { it }
val sub = tensor.asDNArray().V[0]  // DN array of shape (3, 4)
```

<!---END-->

## view function

Reduces dimensionality by fixing one or more axes to a single index.

```kotlin
fun <T, D : Dimension, M : Dimension> MultiArray<T, D>.view(
    index: Int, axis: Int = 0
): MultiArray<T, M>

fun <T, D : Dimension, M : Dimension> MultiArray<T, D>.view(
    indices: IntArray, axes: IntArray
): MultiArray<T, M>
```

Each fixed axis removes one dimension from the result.


<!---FUN view_example-->

```kotlin
val m = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])
val row: MultiArray<Int, D1> = m.view(1)          // [4, 5, 6]
```

<!---END-->

## slice function

Selects a sub-range along one or more axes. Returns a view.

### By range and axis

```kotlin
fun <T, D : Dimension, O : Dimension> MultiArray<T, D>.slice(
    inSlice: ClosedRange<Int>, axis: Int = 0
): NDArray<T, O>
```

<!---FUN slice_by_range_example-->

```kotlin
val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])
val cols = a.slice<Int, D2, D2>(0..1, axis = 1) // columns 0 and 1
```

<!---END-->

### By map of axes

```kotlin
fun <T, D : Dimension, O : Dimension> MultiArray<T, D>.slice(
    indexing: Map<Int, Indexing>
): NDArray<T, O>
```

Keys are axis indices; values are `RInt` (reduces dimension) or `Slice` (keeps dimension).

<!---FUN slice_by_map_example-->

```kotlin
val t = mk.d3array(2, 3, 4) { it }
val s = t.slice<Int, D3, D2>(mapOf(0 to 0.r, 2 to (0..2).toSlice()))
// axis 0 fixed to index 0 (removed), axis 2 sliced to 0..2 (kept)
```

<!---END-->

## Pitfalls

* **Slicing returns a view**, not a copy. Mutating a slice modifies the original array. Call `deepCopy()` to detach.
* **Negative indices are not supported.** Use `shape[axis] - k` explicitly.
* `step` in `Slice` must be **positive**. Reverse iteration is not supported through slicing — use `reversed()` instead.

<seealso style="cards">
<category ref="api-docs">
  <a href="ndarray-class.md" summary="NDArray class — properties, operators, and methods."/>
  <a href="iterating.md" summary="Iterators and collection-style traversal."/>
</category>
<category ref="user-guide">
  <a href="indexing-and-slicing.md" summary="User guide for indexing and slicing with examples."/>
</category>
</seealso>
