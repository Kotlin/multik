# toList

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for toList() — converts an NDArray into a flat Kotlin List, with dimensional
variants for nested lists.
</web-summary>

<card-summary>
Convert an NDArray to a flat List, MutableList, or nested lists by dimension.
</card-summary>

<link-summary>
toList() — convert an NDArray to a Kotlin List.
</link-summary>

## Signatures

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.toList(): List<T>
fun <T, D : Dimension> MultiArray<T, D>.toMutableList(): MutableList<T>

fun <T> MultiArray<T, D2>.toListD2(): List<List<T>>
fun <T> MultiArray<T, D3>.toListD3(): List<List<List<T>>>
fun <T> MultiArray<T, D4>.toListD4(): List<List<List<List<T>>>>
```

**Returns:**

- `toList()` / `toMutableList()`: flat list of all elements in row-major order.
- `toListD2()`, `toListD3()`, `toListD4()`: nested lists matching the array's dimensional structure.

## Example

<!---FUN toList_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
a.toList()       // [1, 2, 3]
a.toMutableList() // mutable [1, 2, 3]

val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
m.toList()       // [1, 2, 3, 4]  (flat)
m.toListD2()     // [[1, 2], [3, 4]]  (nested)
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="toCollection.md" summary="Append elements to a mutable collection."/>
  <a href="toPrimitiveArray.md" summary="Convert to a Kotlin primitive array."/>
  <a href="toArray.md" summary="Convert to nested Kotlin arrays."/>
  <a href="asSequence.md" summary="Lazy sequence over elements."/>
</category>
</seealso>