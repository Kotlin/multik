# scan

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for scan() — returns an NDArray of successive accumulation values, including
the initial value and all intermediate results.
</web-summary>

<card-summary>
Running accumulation that returns all intermediate values as an array.
</card-summary>

<link-summary>
scan() — successive accumulation values as an NDArray.
</link-summary>

Returns an array of successive accumulation values starting from the initial value, applying the
operation to each element from left to right. The result contains the initial value followed by
all intermediate accumulated values.

## Signatures

```kotlin
inline fun <T, D : Dimension, reified R : Any> MultiArray<T, D>.scan(
    initial: R,
    operation: (acc: R, T) -> R
): NDArray<R, D>

inline fun <T, reified R : Any> MultiArray<T, D1>.scanIndexed(
    initial: R,
    operation: (index: Int, acc: R, T) -> R
): D1Array<R>

inline fun <T, D : Dimension, reified R : Any> MultiArray<T, D>.scanMultiIndexed(
    initial: R,
    operation: (index: IntArray, acc: R, T) -> R
): NDArray<R, D>
```

## Parameters

| Parameter   | Type                                | Description                                 |
|-------------|-------------------------------------|---------------------------------------------|
| `initial`   | `R`                                 | Starting accumulator value.                 |
| `operation` | `(acc: R, T) -> R`                  | Combines the accumulator with each element. |
| `operation` | `(index: Int, acc: R, T) -> R`      | With flat index (1D arrays only).           |
| `operation` | `(index: IntArray, acc: R, T) -> R` | With multi-dimensional index.               |

**Returns:** `NDArray<R, D>` containing successive accumulation values.

## Example

<!---FUN scan_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4])

a.scan(0) { acc, v -> acc + v }
// [0, 1, 3, 6, 10]  — cumulative sum with initial 0
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="fold.md" summary="Accumulate to a single final value."/>
  <a href="reduce.md" summary="Accumulate without an initial value."/>
  <a href="sum.md" summary="Sum of all elements."/>
</category>
</seealso>
