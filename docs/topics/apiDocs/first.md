# first

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for first() — returns the first element of an NDArray, or the first matching
a predicate.
</web-summary>

<card-summary>
Get the first element, or the first matching a predicate.
</card-summary>

<link-summary>
first() — first element of an NDArray.
</link-summary>

## Signatures

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.first(): T

inline fun <T, D : Dimension> MultiArray<T, D>.first(
    predicate: (T) -> Boolean
): T

fun <T, D : Dimension> MultiArray<T, D>.firstOrNull(): T?

inline fun <T, D : Dimension> MultiArray<T, D>.firstOrNull(
    predicate: (T) -> Boolean
): T?
```

## Parameters

| Parameter   | Type              | Description                                    |
|-------------|-------------------|------------------------------------------------|
| `predicate` | `(T) -> Boolean`  | Condition to match the first element against.  |

**Returns:**
- `first()`: the first element. Throws `NoSuchElementException` if the array is empty.
- `first(predicate)`: the first matching element. Throws `NoSuchElementException` if none match.
- `firstOrNull()` / `firstOrNull(predicate)`: same as above but returns `null` instead of throwing.

## Example

<!---FUN first_example-->

```kotlin
val a = mk.ndarray(mk[10, 20, 30])

a.first()                // 10
a.first { it > 15 }      // 20
a.firstOrNull { it > 50 } // null
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="last.md" summary="Get the last element."/>
  <a href="find.md" summary="Find the first matching element (returns null)."/>
  <a href="indexOf.md" summary="Find the index of an element."/>
</category>
</seealso>