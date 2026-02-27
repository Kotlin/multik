# indexOf

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for indexOf() — returns the flat index of the first occurrence of an element
in an NDArray, or -1 if not found.
</web-summary>

<card-summary>
Find the flat index of an element's first occurrence, or -1.
</card-summary>

<link-summary>
indexOf() — flat index of the first occurrence of an element.
</link-summary>

Returns the flat (row-major) index of the first occurrence of the specified element, or `-1` if
not found. Predicate-based variants (`indexOfFirst`, `indexOfLast`) and `lastIndexOf` are also
available.

## Signatures

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.indexOf(element: T): Int

inline fun <T, D : Dimension> MultiArray<T, D>.indexOfFirst(
    predicate: (T) -> Boolean
): Int

inline fun <T, D : Dimension> MultiArray<T, D>.indexOfLast(
    predicate: (T) -> Boolean
): Int

fun <T, D : Dimension> MultiArray<T, D>.lastIndexOf(element: T): Int
```

## Parameters

| Parameter   | Type             | Description                          |
|-------------|------------------|--------------------------------------|
| `element`   | `T`              | The element to search for.           |
| `predicate` | `(T) -> Boolean` | Condition to match elements against. |

**Returns:** Flat index (0-based), or `-1` if not found.

## Example

<!---FUN indexOf_example-->

```kotlin
val a = mk.ndarray(mk[10, 20, 30, 20, 10])

a.indexOf(20)                // 1
a.lastIndexOf(20)            // 3
a.indexOfFirst { it > 15 }   // 1
a.indexOfLast { it > 15 }    // 3
a.indexOf(99)                // -1
```

<!---END-->

> The returned index is a **flat** index in row-major order, not a multi-dimensional index.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="contains.md" summary="Check if an element is present."/>
  <a href="find.md" summary="Get the first matching element value."/>
  <a href="first.md" summary="Get the first element."/>
</category>
</seealso>
