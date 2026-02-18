# reduce

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for reduce() — accumulates NDArray elements from left to right, starting
with the first element as the initial accumulator.
</web-summary>

<card-summary>
Accumulate elements from left to right without an initial value.
</card-summary>

<link-summary>
reduce() — left-to-right accumulation starting from the first element.
</link-summary>

## Signatures

```kotlin
inline fun <S, D : Dimension, T : S> MultiArray<T, D>.reduce(
    operation: (acc: S, T) -> S
): S

inline fun <S, T : S> MultiArray<T, D1>.reduceIndexed(
    operation: (index: Int, acc: S, T) -> S
): S

inline fun <S, D : Dimension, T : S> MultiArray<T, D>.reduceMultiIndexed(
    operation: (index: IntArray, acc: S, T) -> S
): S

inline fun <S, D : Dimension, T : S> MultiArray<T, D>.reduceOrNull(
    operation: (acc: S, T) -> S
): S?
```

## Parameters

| Parameter   | Type                                | Description                                 |
|-------------|-------------------------------------|---------------------------------------------|
| `operation` | `(acc: S, T) -> S`                  | Combines the accumulator with each element. |
| `operation` | `(index: Int, acc: S, T) -> S`      | With flat index (1D arrays only).           |
| `operation` | `(index: IntArray, acc: S, T) -> S` | With multi-dimensional index.               |

**Returns:** The final accumulated value. `reduce` throws `UnsupportedOperationException` on empty arrays;
`reduceOrNull` returns `null`.

## Example

<!---FUN reduce_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4])

a.reduce { acc, v -> acc + v }        // 10
a.reduce { acc, v -> acc * v }        // 24

a.reduceIndexed { i, acc, v -> acc + i * v }
// 1 + 1*2 + 2*3 + 3*4 = 21
```

<!---END-->

> `reduce` throws on empty arrays. Use `reduceOrNull` or `fold` with an explicit initial value if
> the array may be empty.
> {style="warning"}

<seealso style="cards">
<category ref="api-docs">
  <a href="fold.md" summary="Accumulate with an explicit initial value."/>
  <a href="scan.md" summary="Accumulate with intermediate results."/>
  <a href="sum.md" summary="Sum of all elements."/>
</category>
</seealso>