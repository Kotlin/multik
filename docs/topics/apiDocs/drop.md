# drop

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for drop() — removes the first n elements from a 1D NDArray.
</web-summary>

<card-summary>
Drop the first n elements from a 1D array.
</card-summary>

<link-summary>
drop() — remove the first n elements of a 1D NDArray.
</link-summary>

## Signatures

```kotlin
fun <T> MultiArray<T, D1>.drop(n: Int): D1Array<T>

inline fun <T> MultiArray<T, D1>.dropWhile(
    predicate: (T) -> Boolean
): NDArray<T, D1>
```

## Parameters

| Parameter   | Type             | Description                                    |
|-------------|------------------|------------------------------------------------|
| `n`         | `Int`            | Number of elements to drop from the beginning. |
| `predicate` | `(T) -> Boolean` | Drop elements while this condition holds.      |

**Returns:** `D1Array<T>` — a new 1D array without the dropped elements.

## Example

<!---FUN drop_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])

a.drop(2)                    // [3, 4, 5]
a.dropWhile { it < 3 }      // [3, 4, 5]
```

<!---END-->

> `drop` and `dropWhile` are only available on **1D** arrays (`MultiArray<T, D1>`).
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="filter.md" summary="Filter elements by a predicate."/>
  <a href="first.md" summary="Get the first element."/>
  <a href="chunked.md" summary="Split into fixed-size chunks."/>
</category>
</seealso>