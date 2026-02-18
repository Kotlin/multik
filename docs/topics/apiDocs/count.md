# count

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for count() — returns the total number of elements or the count of elements
matching a predicate.
</web-summary>

<card-summary>
Count all elements or only those matching a predicate.
</card-summary>

<link-summary>
count() — count NDArray elements, optionally filtered by predicate.
</link-summary>

## Signatures

```kotlin
inline fun <T, D : Dimension> MultiArray<T, D>.count(): Int

inline fun <T, D : Dimension> MultiArray<T, D>.count(
    predicate: (T) -> Boolean
): Int
```

## Parameters

| Parameter   | Type              | Description                            |
|-------------|-------------------|----------------------------------------|
| `predicate` | `(T) -> Boolean`  | Condition to count matching elements.  |

**Returns:**
- Without predicate: total number of elements.
- With predicate: number of elements satisfying `predicate`.

## Example

<!---FUN count_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])

a.count()            // 5
a.count { it > 3 }   // 2
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="all.md" summary="Check if all elements match."/>
  <a href="any.md" summary="Check if any element matches."/>
  <a href="sum.md" summary="Sum of all elements."/>
</category>
</seealso>