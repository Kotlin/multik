# intersect

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for intersect() — returns a set of elements present in both the NDArray
and another collection.
</web-summary>

<card-summary>
Set intersection between an NDArray and another collection.
</card-summary>

<link-summary>
intersect() — elements common to an NDArray and another collection.
</link-summary>

## Signature

```kotlin
infix fun <T, D : Dimension> MultiArray<T, D>.intersect(
    other: Iterable<T>
): Set<T>
```

## Parameters

| Parameter | Type          | Description                       |
|-----------|---------------|-----------------------------------|
| `other`   | `Iterable<T>` | The collection to intersect with. |

**Returns:** `Set<T>` — elements that appear in both the array and `other`.

## Example

<!---FUN intersect_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])

a intersect listOf(3, 4, 5, 6, 7)    // {3, 4, 5}
a intersect listOf(10, 20)           // {}
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="distinct.md" summary="Get unique elements."/>
  <a href="contains.md" summary="Check if an element is present."/>
  <a href="filter.md" summary="Filter elements by a predicate."/>
</category>
</seealso>