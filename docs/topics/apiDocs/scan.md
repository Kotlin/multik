# scan

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for scan() — returns a 1D array of successive accumulation values, including
the initial value and all intermediate results.
</web-summary>

<card-summary>
Running accumulation that returns all intermediate values as an array.
</card-summary>

<link-summary>
scan() — successive accumulation values as a 1D array.
</link-summary>

Returns a 1D array of successive accumulation values starting from the initial value, applying the
operation to each element from left to right. The result contains the initial value followed by
one accumulated value per element, so it has `size + 1` elements. Indexed variants also provide the
element's flat index (`scanIndexed`, 1D only, starting from 0) or multi-dimensional index
(`scanMultiIndexed`). The result element type `R` must be a supported `DataType`.

## Signatures

```kotlin
inline fun <T, D : Dimension, reified R : Any> MultiArray<T, D>.scan(
    initial: R,
    operation: (acc: R, T) -> R
): D1Array<R>

inline fun <T, reified R : Any> MultiArray<T, D1>.scanIndexed(
    initial: R,
    operation: (index: Int, acc: R, T) -> R
): D1Array<R>

inline fun <T, D : Dimension, reified R : Any> MultiArray<T, D>.scanMultiIndexed(
    initial: R,
    operation: (index: IntArray, acc: R, T) -> R
): D1Array<R>
```

## Parameters

| Parameter   | Type                                | Description                                                     |
|-------------|-------------------------------------|-----------------------------------------------------------------|
| `initial`   | `R`                                 | Starting accumulator value and the first element of the result. |
| `operation` | `(acc: R, T) -> R`                  | Combines the accumulator with each element.                     |
| `operation` | `(index: Int, acc: R, T) -> R`      | With flat index from 0 (1D arrays only).                        |
| `operation` | `(index: IntArray, acc: R, T) -> R` | With multi-dimensional index.                                   |

**Returns:** `D1Array<R>` — a new array of `size + 1` values: `initial`, then the accumulated values
in row-major order.

## Example

<!---FUN scan_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4])

a.scan(0) { acc, v -> acc + v }
// [0, 1, 3, 6, 10]  — cumulative sum with initial 0

a.scanIndexed(0) { i, acc, v -> acc + i * v }
// [0, 0, 2, 8, 20]  — indices start at 0

val b = mk.ndarray(mk[mk[1, 2], mk[3, 4]])

b.scan(0) { acc, v -> acc + v }
// [0, 1, 3, 6, 10]  — always 1D, in row-major order

b.scanMultiIndexed(0) { idx, acc, v -> if (idx[0] == idx[1]) acc + v else acc }
// [0, 1, 1, 1, 5]  — running sum of the diagonal
```

<!---END-->

> `scan` always returns a **1D** array with one more element than the source, whatever the source shape.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="fold.md" summary="Accumulate to a single final value."/>
  <a href="reduce.md" summary="Accumulate without an initial value."/>
  <a href="sum.md" summary="Sum of all elements."/>
</category>
</seealso>
