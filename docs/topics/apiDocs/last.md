# last

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for last() — returns the last element of an NDArray, or the last matching
a predicate.
</web-summary>

<card-summary>
Get the last element, or the last matching a predicate.
</card-summary>

<link-summary>
last() — last element of an NDArray.
</link-summary>

## Signatures

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.last(): T

inline fun <T, D : Dimension> MultiArray<T, D>.last(
    predicate: (T) -> Boolean
): T

fun <T, D : Dimension> MultiArray<T, D>.lastOrNull(): T?

inline fun <T, D : Dimension> MultiArray<T, D>.lastOrNull(
    predicate: (T) -> Boolean
): T?
```

## Parameters

| Parameter   | Type             | Description                                  |
|-------------|------------------|----------------------------------------------|
| `predicate` | `(T) -> Boolean` | Condition to match the last element against. |

**Returns:**

- `last()`: the last element. Throws `NoSuchElementException` if the array is empty.
- `last(predicate)`: the last matching element. Throws `NoSuchElementException` if none match.
- `lastOrNull()` / `lastOrNull(predicate)`: same as above but returns `null` instead of throwing.

## Example

<!---FUN last_example-->

```kotlin
val a = mk.ndarray(mk[10, 20, 30])

a.last()                 // 30
a.last { it < 25 }       // 20
a.lastOrNull { it > 50 } // null
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="first.md" summary="Get the first element."/>
  <a href="find.md" summary="Find the first matching element (returns null)."/>
  <a href="indexOf.md" summary="Find the index of an element."/>
</category>
</seealso>