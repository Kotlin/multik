# Iterating

<!---IMPORT samples.docs.apiDocs.ArrayObjects-->

<web-summary>
API reference for iterating over Multik NDArrays: NDArrayIterator, MultiIndexProgression,
and collection-style operations like forEach, map, filter, fold, and reduce.
</web-summary>

<card-summary>
NDArrayIterator, MultiIndexProgression, and collection-style operations on NDArrays.
</card-summary>

<link-summary>
Iteration API — NDArrayIterator, multi-index progression, and collection operations.
</link-summary>

## NDArrayIterator

Traverses elements of a non-consistent array in logical (row-major) order, respecting `offset`, `strides`, and `shape`.

```kotlin
public class NDArrayIterator<T>(
    private val data: MemoryView<T>,
    private val offset: Int = 0,
    private val strides: IntArray,
    private val shape: IntArray
) : Iterator<T>
```

| Member               | Description                                         |
|----------------------|-----------------------------------------------------|
| `index: IntArray`    | Current multi-dimensional index.                    |
| `hasNext(): Boolean` | `true` until all index positions are exhausted.     |
| `next(): T`          | Returns the current element and advances the index. |

For **consistent** arrays (contiguous, default strides), `NDArray.iterator()` delegates directly to
`MemoryView.iterator()` for better performance.

## MultiIndexProgression

Iterates over all valid multi-dimensional index tuples within a shape.

```kotlin
public class MultiIndexProgression(
    public val first: IntArray,
    public val last: IntArray,
    public val step: Int = 1
)
```

| Member       | Description                                                  |
|--------------|--------------------------------------------------------------|
| `first`      | Start indices (inclusive).                                   |
| `last`       | End indices (inclusive).                                     |
| `step`       | Traversal step. Positive for forward, negative for backward. |
| `reverse`    | Lazy reversed progression.                                   |
| `iterator()` | Returns an `Iterator<IntArray>`.                             |

### Creating progressions

<!---FUN creating_progression_example-->

```kotlin
val start = intArrayOf(0, 0)
val end = intArrayOf(1, 2)

val prog = start..end                    // forward, step 1
val stepped = (start..end).step(2)       // forward, step 2
val rev = end downTo start               // backward, step 1
```

<!---END-->

### forEach on progressions

<!---FUN forEach_progression_example-->

```kotlin
val m = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])
m.multiIndices.forEach { idx ->
    // idx is IntArray, e.g. [0, 0], [0, 1], ...
}
```

<!---END-->

## Collection-style operations

All operations below are extension functions on `MultiArray<T, D>`.

### Predicates

| Function   | Signature                              | Description                     |
|------------|----------------------------------------|---------------------------------|
| `all`      | `(predicate: (T) -> Boolean): Boolean` | `true` if all elements match.   |
| `any`      | `(predicate: (T) -> Boolean): Boolean` | `true` if at least one matches. |
| `contains` | `(element: T): Boolean`                | `true` if element is present.   |

### Traversal

| Function              | Signature                         | Description                     |
|-----------------------|-----------------------------------|---------------------------------|
| `forEach`             | `(action: (T) -> Unit)`           | Applies action to each element. |
| `forEachIndexed`      | `(action: (Int, T) -> Unit)`      | With flat index.                |
| `forEachMultiIndexed` | `(action: (IntArray, T) -> Unit)` | With multi-dimensional index.   |

### Transformation

| Function          | Signature                                        | Description                              |
|-------------------|--------------------------------------------------|------------------------------------------|
| `map`             | `(transform: (T) -> R): NDArray<R, D>`           | Element-wise transform, preserves shape. |
| `mapIndexed`      | `(transform: (Int, T) -> R): D1Array<R>`         | With flat index, returns 1D.             |
| `mapMultiIndexed` | `(transform: (IntArray, T) -> R): NDArray<R, D>` | With multi-index, preserves shape.       |

### Filtering

| Function             | Signature                                           | Description                            |
|----------------------|-----------------------------------------------------|----------------------------------------|
| `filter`             | `(predicate: (T) -> Boolean): D1Array<T>`           | Returns 1D array of matching elements. |
| `filterIndexed`      | `(predicate: (Int, T) -> Boolean): D1Array<T>`      | With flat index.                       |
| `filterMultiIndexed` | `(predicate: (IntArray, T) -> Boolean): D1Array<T>` | With multi-dimensional index.          |

### Aggregation

| Function      | Signature                               | Description                                        |
|---------------|-----------------------------------------|----------------------------------------------------|
| `fold`        | `(initial: R, op: (R, T) -> R): R`      | Accumulate from initial value.                     |
| `foldIndexed` | `(initial: R, op: (Int, R, T) -> R): R` | With flat index.                                   |
| `reduce`      | `(op: (T, T) -> T): T`                  | Accumulate without initial value. Throws on empty. |
| `sum`         | `(): T`                                 | Sum of all elements.                               |
| `max`         | `(): T?`                                | Maximum, or `null` if empty.                       |
| `min`         | `(): T?`                                | Minimum, or `null` if empty.                       |

### Search

| Function  | Signature           | Description                                              |
|-----------|---------------------|----------------------------------------------------------|
| `first`   | `(): T`             | First element. Throws `NoSuchElementException` if empty. |
| `last`    | `(): T`             | Last element. Throws `NoSuchElementException` if empty.  |
| `indexOf` | `(element: T): Int` | Flat index of the first occurrence, or `-1`.             |

### Ordering

| Function     | Returns         | Description                               |
|--------------|-----------------|-------------------------------------------|
| `sorted()`   | `NDArray<T, D>` | New array with elements sorted.           |
| `reversed()` | `NDArray<T, D>` | New array with elements in reverse order. |
| `distinct()` | `D1Array<T>`    | Unique elements as a 1D array.            |

### Partitioning

| Function    | Signature                                                   | Description                          |
|-------------|-------------------------------------------------------------|--------------------------------------|
| `partition` | `(predicate: (T) -> Boolean): Pair<D1Array<T>, D1Array<T>>` | Split into (matching, non-matching). |
| `chunked`   | `(size: Int): D2Array<T>`                                   | Split into fixed-size chunks.        |
| `windowed`  | `(size: Int, step: Int, limit: Boolean): D2Array<T>`        | Sliding window.                      |

### Conversion

| Function          | Returns          | Description                  |
|-------------------|------------------|------------------------------|
| `toList()`        | `List<T>`        | Flat list of all elements.   |
| `toMutableList()` | `MutableList<T>` | Mutable flat list.           |
| `toSet()`         | `Set<T>`         | Set of unique elements.      |
| `toMutableSet()`  | `MutableSet<T>`  | Mutable set.                 |
| `asSequence()`    | `Sequence<T>`    | Lazy sequence over elements. |

### Logic

| Function  | Signature                                    | Description                     |
|-----------|----------------------------------------------|---------------------------------|
| `and`     | `(other: MultiArray<T, D>): NDArray<Int, D>` | Element-wise logical AND (1/0). |
| `or`      | `(other: MultiArray<T, D>): NDArray<Int, D>` | Element-wise logical OR (1/0).  |
| `maximum` | `(other: MultiArray<T, D>): NDArray<T, D>`   | Element-wise max.               |
| `minimum` | `(other: MultiArray<T, D>): NDArray<T, D>`   | Element-wise min.               |

<seealso style="cards">
<category ref="api-docs">
  <a href="ndarray-class.md" summary="NDArray class — properties, operators, and methods."/>
  <a href="indexing-routines.md" summary="Slice, RInt, and view helpers."/>
</category>
<category ref="user-guide">
  <a href="iterating-over-arrays.md" summary="User guide for iterating with examples."/>
</category>
</seealso>
