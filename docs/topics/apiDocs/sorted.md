# sorted

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for sorted() — returns a new NDArray with elements sorted in ascending order.
</web-summary>

<card-summary>
Return a new array with elements sorted in ascending order.
</card-summary>

<link-summary>
sorted() — sort NDArray elements in ascending order.
</link-summary>

## Signature

```kotlin
fun <T : Number, D : Dimension> MultiArray<T, D>.sorted(): NDArray<T, D>
```

**Returns:** `NDArray<T, D>` — a new array with the same shape, elements sorted in ascending row-major order.

## Example

<!---FUN sorted_example-->

```kotlin
val a = mk.ndarray(mk[3, 1, 4, 1, 5, 9])
a.sorted()  // [1, 1, 3, 4, 5, 9]

val m = mk.ndarray(mk[mk[3, 1], mk[4, 2]])
m.sorted()
// [[1, 2],
//  [3, 4]]
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="reversed.md" summary="Reverse element order."/>
  <a href="distinct.md" summary="Get unique elements."/>
  <a href="max.md" summary="Find the largest element."/>
  <a href="min.md" summary="Find the smallest element."/>
</category>
</seealso>