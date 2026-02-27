# reversed

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for reversed() — returns a new NDArray with elements in reverse order.
</web-summary>

<card-summary>
Return a new array with elements in reverse order.
</card-summary>

<link-summary>
reversed() — reverse the element order of an NDArray.
</link-summary>

Returns a new array with elements in reverse row-major order. The shape is preserved.

## Signature

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.reversed(): NDArray<T, D>
```

**Returns:** `NDArray<T, D>` — a new array with the same shape but elements in reverse row-major order.

## Example

<!---FUN reversed_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])
a.reversed()  // [5, 4, 3, 2, 1]

val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
m.reversed()
// [[4, 3],
//  [2, 1]]
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="sorted.md" summary="Sort elements."/>
  <a href="distinct.md" summary="Get unique elements."/>
</category>
</seealso>
