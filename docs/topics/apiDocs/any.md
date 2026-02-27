# any

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for any() — returns true if an NDArray has at least one element, or if at least
one element matches a predicate.
</web-summary>

<card-summary>
Returns true if at least one element exists or matches a predicate.
</card-summary>

<link-summary>
any() — check if any NDArray element satisfies a condition.
</link-summary>

Tests whether the array is non-empty (no-arg form) or whether at least one element satisfies the
given predicate. Returns `false` for empty arrays when a predicate is provided.

## Signatures

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.any(): Boolean

inline fun <T, D : Dimension> MultiArray<T, D>.any(
    predicate: (T) -> Boolean
): Boolean
```

## Parameters

| Parameter   | Type             | Description                             |
|-------------|------------------|-----------------------------------------|
| `predicate` | `(T) -> Boolean` | Condition to test each element against. |

**Returns:**

- Without predicate: `true` if the array is not empty.
- With predicate: `true` if at least one element satisfies `predicate`. Returns `false` for empty arrays.

## Example

<!---FUN any_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4])

a.any()              // true
a.any { it > 3 }     // true
a.any { it > 10 }    // false
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="all.md" summary="Check if all elements match a predicate."/>
  <a href="contains.md" summary="Check if an element is present."/>
  <a href="find.md" summary="Find the first matching element."/>
</category>
</seealso>
