# repeat

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for repeat() — tiles the array's elements n times into a new flat 1D array.
</web-summary>

<card-summary>
Tile array elements n times into a flat 1D array.
</card-summary>

<link-summary>
repeat() — repeat NDArray elements n times.
</link-summary>

Tiles the array's elements `n` times into a new flat 1D array. Multi-dimensional arrays are
flattened before tiling.

## Signature

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.repeat(n: Int): D1Array<T>
```

## Parameters

| Parameter | Type  | Description                                      |
|-----------|-------|--------------------------------------------------|
| `n`       | `Int` | Number of repetitions. Must be at least 1.       |

**Returns:** `D1Array<T>` — a new flat 1D array with size `this.size * n`.

## Example

<!---FUN repeat_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
a.repeat(3) // [1, 2, 3, 1, 2, 3, 1, 2, 3]

val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
m.repeat(2) // [1, 2, 3, 4, 1, 2, 3, 4]
```

<!---END-->

> `repeat` always **flattens** the array before tiling. The result is always 1D regardless of the
> input dimensionality.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="append.md" summary="Append elements or arrays."/>
  <a href="windowed.md" summary="Sliding window over elements."/>
</category>
</seealso>
