# chunked

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for chunked() — splits an NDArray into fixed-size chunks, returned as a 2D array.
</web-summary>

<card-summary>
Split an NDArray into fixed-size chunks as a 2D array.
</card-summary>

<link-summary>
chunked() — split an NDArray into fixed-size chunks.
</link-summary>

## Signature

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.chunked(size: Int): NDArray<T, D2>
```

## Parameters

| Parameter | Type  | Description                       |
|-----------|-------|-----------------------------------|
| `size`    | `Int` | Number of elements in each chunk. |

**Returns:** `NDArray<T, D2>` — a 2D array where each row is a chunk. If the total number of elements
is not evenly divisible by `size`, the last row is padded with zeros.

## Example

<!---FUN chunked_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])

val c = a.chunked(2)
// [[1, 2],
//  [3, 4],
//  [5, 0]]   — last chunk padded with 0
```

<!---END-->

## Pitfalls

* The last chunk is **zero-padded** if elements don't divide evenly. Check the original size if this matters.

<seealso style="cards">
<category ref="api-docs">
  <a href="windowed.md" summary="Sliding window over elements."/>
  <a href="partition.md" summary="Split into matching and non-matching."/>
  <a href="drop.md" summary="Drop the first n elements."/>
</category>
</seealso>