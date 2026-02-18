# min

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for min() — returns the smallest element in an NDArray, or null if empty.
</web-summary>

<card-summary>
Find the smallest element, or null if the array is empty.
</card-summary>

<link-summary>
min() — smallest element of an NDArray.
</link-summary>

## Signatures

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.min(): T?
        where T : Number, T : Comparable<T>

inline fun <T, D : Dimension, R : Comparable<R>> MultiArray<T, D>.minBy(
    selector: (T) -> R
): T?

fun <T, D : Dimension> MultiArray<T, D>.minWith(
    comparator: Comparator<in T>
): T?
```

## Parameters

| Parameter    | Type               | Description                                          |
|--------------|--------------------|------------------------------------------------------|
| `selector`   | `(T) -> R`         | Maps each element to a comparable value for ranking. |
| `comparator` | `Comparator<in T>` | Custom comparator for ordering.                      |

**Returns:** The smallest element, or `null` if the array is empty.

## Example

<!---FUN min_example-->

```kotlin
val a = mk.ndarray(mk[3, 1, 4, 1, 5])

a.min()                        // 1
a.minBy { -it }                // 5  (smallest by negated value)
a.minWith(compareBy { it })    // 1
```

<!---END-->

> For element-wise minimum between two arrays, see [`minimum()`](minimum.md).
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="max.md" summary="Find the largest element."/>
  <a href="minimum.md" summary="Element-wise minimum of two arrays."/>
  <a href="sum.md" summary="Sum of all elements."/>
  <a href="average.md" summary="Mean of all elements."/>
</category>
</seealso>