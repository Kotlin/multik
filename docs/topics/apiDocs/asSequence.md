# asSequence

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for asSequence() — wraps an NDArray into a lazy Kotlin Sequence.
</web-summary>

<card-summary>
Wraps an NDArray into a lazy Sequence for on-demand iteration.
</card-summary>

<link-summary>
asSequence() — create a lazy Sequence from an NDArray.
</link-summary>

## Signature

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.asSequence(): Sequence<T>
```

**Returns:** A `Sequence<T>` that yields elements lazily in row-major order.

## Example

<!---FUN asSequence_example-->

```kotlin
val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

val result = a.asSequence()
    .filter { it > 2 }
    .map { it * 10 }
    .toList()
// [30, 40, 50, 60]
```

<!---END-->

Use `asSequence()` when chaining multiple operations to avoid creating intermediate arrays.

<seealso style="cards">
<category ref="api-docs">
  <a href="toList.md" summary="Convert an NDArray to a flat List."/>
  <a href="forEach.md" summary="Apply an action to each element."/>
  <a href="map.md" summary="Transform each element."/>
</category>
</seealso>