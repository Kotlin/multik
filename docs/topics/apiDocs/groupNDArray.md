# groupNDArray

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for groupNDArrayBy() — groups NDArray elements by a key function into a map
of 1D NDArrays.
</web-summary>

<card-summary>
Group elements by a key function into a map of 1D NDArrays.
</card-summary>

<link-summary>
groupNDArrayBy() — group NDArray elements by key.
</link-summary>

## Signatures

```kotlin
inline fun <T, D : Dimension, K> MultiArray<T, D>.groupNDArrayBy(
    keySelector: (T) -> K
): Map<K, NDArray<T, D1>>

inline fun <T, D : Dimension, K, V : Number> MultiArray<T, D>.groupNDArrayBy(
    keySelector: (T) -> K,
    valueTransform: (T) -> V
): Map<K, NDArray<V, D1>>

inline fun <T, D : Dimension, K> MultiArray<T, D>.groupingNDArrayBy(
    keySelector: (T) -> K
): Grouping<T, K>
```

## Parameters

| Parameter        | Type       | Description                                        |
|------------------|------------|----------------------------------------------------|
| `keySelector`    | `(T) -> K` | Extracts the group key from each element.          |
| `valueTransform` | `(T) -> V` | Transforms the element before adding to the group. |

**Returns:** `Map<K, NDArray<T, D1>>` — each key maps to a 1D array of matching elements.

## Example

<!---FUN groupNDArrayBy_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6])

val groups = a.groupNDArrayBy { if (it % 2 == 0) "even" else "odd" }
// "odd"  -> [1, 3, 5]
// "even" -> [2, 4, 6]

val grouping = a.groupingNDArrayBy { it % 3 }
grouping.eachCount()  // {1=2, 2=2, 0=2}
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="associate.md" summary="Build a map from key-value pairs."/>
  <a href="partition.md" summary="Split into two arrays by predicate."/>
  <a href="filter.md" summary="Filter elements by a predicate."/>
</category>
</seealso>