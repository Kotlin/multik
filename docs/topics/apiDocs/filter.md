# filter

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for filter() — returns a 1D NDArray containing only elements that match a predicate.
</web-summary>

<card-summary>
Filter elements by a predicate, returning a 1D array of matches.
</card-summary>

<link-summary>
filter() — select elements matching a predicate.
</link-summary>

## Signatures

```kotlin
inline fun <T, D : Dimension> MultiArray<T, D>.filter(
    predicate: (T) -> Boolean
): D1Array<T>

inline fun <T> MultiArray<T, D1>.filterIndexed(
    predicate: (index: Int, T) -> Boolean
): D1Array<T>

inline fun <T, D : Dimension> MultiArray<T, D>.filterMultiIndexed(
    predicate: (index: IntArray, T) -> Boolean
): D1Array<T>

inline fun <T, D : Dimension> MultiArray<T, D>.filterNot(
    predicate: (T) -> Boolean
): D1Array<T>
```

## Parameters

| Parameter   | Type                              | Description                                      |
|-------------|-----------------------------------|--------------------------------------------------|
| `predicate` | `(T) -> Boolean`                  | Condition to match.                              |
| `predicate` | `(index: Int, T) -> Boolean`      | Condition with flat index (1D arrays only).      |
| `predicate` | `(index: IntArray, T) -> Boolean` | Condition with multi-dimensional index.          |

**Returns:** `D1Array<T>` — a flat 1D array of matching elements.

## Example

<!---FUN filter_example-->

```kotlin
val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

a.filter { it > 3 }                       // [4, 5, 6]
a.filterNot { it > 3 }                    // [1, 2, 3]

val b = mk.ndarray(mk[10, 20, 30, 40])
b.filterIndexed { i, v -> i % 2 == 0 }    // [10, 30]
```

<!---END-->

> `filter` always returns a **1D** array, regardless of the input dimensionality.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="find.md" summary="Find the first matching element."/>
  <a href="map.md" summary="Transform each element."/>
  <a href="partition.md" summary="Split into matching and non-matching."/>
  <a href="distinct.md" summary="Get unique elements."/>
</category>
</seealso>