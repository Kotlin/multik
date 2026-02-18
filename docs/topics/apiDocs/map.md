# map

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for map() — transforms each element of an NDArray, returning a new array
of the same shape.
</web-summary>

<card-summary>
Transform each element, returning a new array of the same shape.
</card-summary>

<link-summary>
map() — element-wise transformation preserving shape.
</link-summary>

## Signatures

```kotlin
inline fun <T, D : Dimension, reified R : Any> MultiArray<T, D>.map(
    transform: (T) -> R
): NDArray<R, D>

inline fun <T, reified R : Any> MultiArray<T, D1>.mapIndexed(
    transform: (index: Int, T) -> R
): D1Array<R>

inline fun <T, D : Dimension, reified R : Any> MultiArray<T, D>.mapMultiIndexed(
    transform: (index: IntArray, T) -> R
): NDArray<R, D>

inline fun <T, D : Dimension, reified R : Any> MultiArray<T, D>.mapNotNull(
    transform: (T) -> R?
): NDArray<R, D>
```

## Parameters

| Parameter   | Type                        | Description                       |
|-------------|-----------------------------|-----------------------------------|
| `transform` | `(T) -> R`                  | Maps each element to a new value. |
| `transform` | `(index: Int, T) -> R`      | With flat index (1D arrays only). |
| `transform` | `(index: IntArray, T) -> R` | With multi-dimensional index.     |

**Returns:** `NDArray<R, D>` — a new array with the same shape and transformed elements.

## Example

```kotlin
val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])

val doubled = a.map { it * 2 }
// [[2, 4],
//  [6, 8]]

val b = mk.ndarray(mk[10, 20, 30])
b.mapIndexed { i, v -> "$i:$v" }
// ["0:10", "1:20", "2:30"]

a.mapMultiIndexed { idx, v -> idx.sum() + v }
// [[1, 3],
//  [4, 6]]
```

> `map` preserves the **shape** of the original array. Use `flatMap` if you need to change the shape.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="flatMap.md" summary="Transform and flatten to 1D."/>
  <a href="filter.md" summary="Select elements by a predicate."/>
  <a href="forEach.md" summary="Apply an action without returning a new array."/>
  <a href="onEach.md" summary="Apply an action and return the same array."/>
</category>
</seealso>