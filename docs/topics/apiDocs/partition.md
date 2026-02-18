# partition

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for partition() — splits an NDArray into two 1D arrays: elements matching
a predicate and elements not matching.
</web-summary>

<card-summary>
Split elements into two 1D arrays: matching and non-matching.
</card-summary>

<link-summary>
partition() — split by predicate into (matching, non-matching).
</link-summary>

## Signature

```kotlin
inline fun <T, D : Dimension> MultiArray<T, D>.partition(
    predicate: (T) -> Boolean
): Pair<NDArray<T, D1>, NDArray<T, D1>>
```

## Parameters

| Parameter   | Type             | Description                         |
|-------------|------------------|-------------------------------------|
| `predicate` | `(T) -> Boolean` | Condition to classify each element. |

**Returns:** `Pair<NDArray<T, D1>, NDArray<T, D1>>` — `first` contains elements where the predicate
is `true`, `second` contains the rest.

## Example

<!---FUN partition_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6])

val (evens, odds) = a.partition { it % 2 == 0 }
// evens: [2, 4, 6]
// odds:  [1, 3, 5]
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="filter.md" summary="Filter elements by a predicate."/>
  <a href="groupNDArray.md" summary="Group by key into multiple arrays."/>
  <a href="chunked.md" summary="Split into fixed-size chunks."/>
</category>
</seealso>