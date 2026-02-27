# all

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for all() — returns true if all elements of an NDArray match a given predicate.
</web-summary>

<card-summary>
Returns true if all elements match a predicate. Empty arrays return true.
</card-summary>

<link-summary>
all() — check if all NDArray elements satisfy a predicate.
</link-summary>

Tests whether every element in the array satisfies the given predicate. Returns `true` for
empty arrays (vacuous truth), consistent with Kotlin's `Iterable.all`.

## Signature

```kotlin
inline fun <T, D : Dimension> MultiArray<T, D>.all(
    predicate: (T) -> Boolean
): Boolean
```

## Parameters

| Parameter   | Type             | Description                             |
|-------------|------------------|-----------------------------------------|
| `predicate` | `(T) -> Boolean` | Condition to test each element against. |

**Returns:** `true` if every element satisfies `predicate`. Returns `true` for empty arrays.

## Example

<!---FUN all_example-->

```kotlin
val a = mk.ndarray(mk[2, 4, 6, 8])

a.all { it % 2 == 0 }   // true
a.all { it > 5 }         // false

val empty = mk.ndarray(mk[0]).drop(1)
empty.all { it > 0 }     // true (vacuously true)
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="any.md" summary="Check if any element matches a predicate."/>
  <a href="filter.md" summary="Filter elements by a predicate."/>
  <a href="count.md" summary="Count elements matching a predicate."/>
</category>
</seealso>
