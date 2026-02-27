# find

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for find() — returns the first element matching a predicate, or null.
</web-summary>

<card-summary>
Find the first element matching a predicate, or null if none found.
</card-summary>

<link-summary>
find() — first element matching a predicate, or null.
</link-summary>

Returns the first element matching the predicate, or `null` if no match is found. `findLast`
searches from the end of the array.

## Signatures

```kotlin
inline fun <T, D : Dimension> MultiArray<T, D>.find(
    predicate: (T) -> Boolean
): T?

inline fun <T, D : Dimension> MultiArray<T, D>.findLast(
    predicate: (T) -> Boolean
): T?
```

## Parameters

| Parameter   | Type             | Description                             |
|-------------|------------------|-----------------------------------------|
| `predicate` | `(T) -> Boolean` | Condition to test each element against. |

**Returns:** The first (or last) matching element, or `null` if no element matches.

## Example

<!---FUN find_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])

a.find { it > 3 }       // 4
a.findLast { it > 3 }   // 5
a.find { it > 10 }      // null
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="first.md" summary="Get the first element (throws if empty)."/>
  <a href="indexOf.md" summary="Get the index of an element."/>
  <a href="filter.md" summary="Filter all matching elements."/>
</category>
</seealso>
