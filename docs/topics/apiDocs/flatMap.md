# flatMap

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for flatMap() — transforms each NDArray element into an iterable and flattens
the results into a 1D array.
</web-summary>

<card-summary>
Transform each element into an iterable and flatten to a 1D array.
</card-summary>

<link-summary>
flatMap() — transform and flatten NDArray elements.
</link-summary>

## Signatures

```kotlin
inline fun <T, D : Dimension, reified R> MultiArray<T, D>.flatMap(
    transform: (T) -> Iterable<R>
): D1Array<R>

inline fun <T, reified R> MultiArray<T, D1>.flatMapIndexed(
    transform: (index: Int, T) -> Iterable<R>
): D1Array<R>

inline fun <T, reified R> MultiArray<T, D1>.flatMapMultiIndexed(
    transform: (index: IntArray, T) -> Iterable<R>
): D1Array<R>
```

## Parameters

| Parameter   | Type                                  | Description                                  |
|-------------|---------------------------------------|----------------------------------------------|
| `transform` | `(T) -> Iterable<R>`                  | Maps each element to an iterable of results. |
| `transform` | `(index: Int, T) -> Iterable<R>`      | With flat index (1D arrays only).            |
| `transform` | `(index: IntArray, T) -> Iterable<R>` | With multi-dimensional index.                |

**Returns:** `D1Array<R>` — all results flattened into a single 1D array.

## Example

<!---FUN flatMap_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])

val result = a.flatMap { listOf(it, it * 10) }
// [1, 10, 2, 20, 3, 30]
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="map.md" summary="Transform each element (preserves shape)."/>
  <a href="filter.md" summary="Filter elements by a predicate."/>
  <a href="fold.md" summary="Accumulate elements from an initial value."/>
</category>
</seealso>