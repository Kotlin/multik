# contains

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for contains() — checks if an NDArray contains a specific element.
</web-summary>

<card-summary>
Check if an element is present in an NDArray. Supports the `in` operator.
</card-summary>

<link-summary>
contains() — check if an NDArray contains a given element.
</link-summary>

## Signature

```kotlin
operator fun <T, D : Dimension> MultiArray<T, D>.contains(element: T): Boolean
```

## Parameters

| Parameter | Type | Description                |
|-----------|------|----------------------------|
| `element` | `T`  | The element to search for. |

**Returns:** `true` if the element is found anywhere in the array.

## Example

<!---FUN contains_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])

a.contains(3)    // true
3 in a           // true  (operator syntax)
10 in a          // false
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="indexOf.md" summary="Find the index of an element."/>
  <a href="find.md" summary="Find the first element matching a predicate."/>
  <a href="any.md" summary="Check if any element matches a predicate."/>
</category>
</seealso>