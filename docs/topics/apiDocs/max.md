# max

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for max() — returns the largest element in an NDArray, or null if empty.
</web-summary>

<card-summary>
Find the largest element, or null if the array is empty.
</card-summary>

<link-summary>
max() — largest element of an NDArray.
</link-summary>

Returns the largest element in the array, or `null` if the array is empty. `maxBy` and `maxWith`
variants allow custom comparison via a selector function or a `Comparator`.

## Signatures

```kotlin
fun <T : Number, D : Dimension> MultiArray<T, D>.max(): T?

inline fun <T, D : Dimension, R : Comparable<R>> MultiArray<T, D>.maxBy(
    selector: (T) -> R
): T?

fun <T, D : Dimension> MultiArray<T, D>.maxWith(
    comparator: Comparator<in T>
): T?
```

## Parameters

| Parameter    | Type               | Description                                          |
|--------------|--------------------|------------------------------------------------------|
| `selector`   | `(T) -> R`         | Maps each element to a comparable value for ranking. |
| `comparator` | `Comparator<in T>` | Custom comparator for ordering.                      |

**Returns:** The largest element, or `null` if the array is empty.

## Example

<!---FUN max_example-->

```kotlin
val a = mk.ndarray(mk[3, 1, 4, 1, 5])

a.max()                        // 5
a.maxBy { -it }                // 1  (largest by negated value)
a.maxWith(compareBy { it })    // 5
```

<!---END-->

> For element-wise maximum between two arrays, see [`maximum()`](maximum.md).
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="min.md" summary="Find the smallest element."/>
  <a href="maximum.md" summary="Element-wise maximum of two arrays."/>
  <a href="sum.md" summary="Sum of all elements."/>
  <a href="average.md" summary="Mean of all elements."/>
</category>
</seealso>
