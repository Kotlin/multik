# distinct

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for distinct() — returns a 1D NDArray containing only the unique elements.
</web-summary>

<card-summary>
Returns a 1D array of unique elements, preserving first-occurrence order.
</card-summary>

<link-summary>
distinct() — unique elements of an NDArray as a 1D array.
</link-summary>

Returns a 1D array containing only the unique elements, preserving first-occurrence order.
`distinctBy` compares elements by the key returned from the selector function.

## Signatures

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.distinct(): NDArray<T, D1>

inline fun <T, D : Dimension, K> MultiArray<T, D>.distinctBy(
    selector: (T) -> K
): NDArray<T, D1>
```

## Parameters

| Parameter  | Type       | Description                                          |
|------------|------------|------------------------------------------------------|
| `selector` | `(T) -> K` | Key function — elements with the same key are duplicates.|

**Returns:** `NDArray<T, D1>` — a 1D array of unique elements in first-occurrence order.

## Example

<!---FUN distinct_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 2, 3, 1, 4])

a.distinct()                  // [1, 2, 3, 4]
a.distinctBy { it % 2 }      // [1, 2]  (first odd, first even)
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="filter.md" summary="Filter elements by a predicate."/>
  <a href="intersect.md" summary="Set intersection with another collection."/>
  <a href="sorted.md" summary="Sort elements."/>
</category>
</seealso>
