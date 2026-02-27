# fold

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for fold() — accumulates a value starting from an initial value, applying an
operation to each NDArray element from left to right.
</web-summary>

<card-summary>
Accumulate elements from left to right with an initial value.
</card-summary>

<link-summary>
fold() — left-to-right accumulation with an initial value.
</link-summary>

Accumulates a value by applying the operation to each element from left to right, starting from
the given initial value. The result type `R` can differ from the element type, making `fold`
suitable for building strings, collections, or other aggregate values from array elements.

## Signatures

```kotlin
inline fun <T, D : Dimension, R> MultiArray<T, D>.fold(
    initial: R,
    operation: (acc: R, T) -> R
): R

inline fun <T, R> MultiArray<T, D1>.foldIndexed(
    initial: R,
    operation: (index: Int, acc: R, T) -> R
): R

inline fun <T, D : Dimension, R> MultiArray<T, D>.foldMultiIndexed(
    initial: R,
    operation: (index: IntArray, acc: R, T) -> R
): R
```

## Parameters

| Parameter   | Type                                | Description                                 |
|-------------|-------------------------------------|---------------------------------------------|
| `initial`   | `R`                                 | Starting accumulator value.                 |
| `operation` | `(acc: R, T) -> R`                  | Combines the accumulator with each element. |
| `operation` | `(index: Int, acc: R, T) -> R`      | With flat index (1D arrays only).           |
| `operation` | `(index: IntArray, acc: R, T) -> R` | With multi-dimensional index.               |

**Returns:** The final accumulated value of type `R`.

## Example

<!---FUN fold_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4])

a.fold(0) { acc, v -> acc + v }                // 10
a.fold("") { acc, v -> "$acc$v" }              // "1234"

val b = mk.ndarray(mk[10, 20, 30])
b.foldIndexed(0) { i, acc, v -> acc + i * v }  // 0*10 + 1*20 + 2*30 = 80
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="reduce.md" summary="Accumulate without an initial value."/>
  <a href="scan.md" summary="Accumulate with intermediate results."/>
  <a href="sum.md" summary="Sum of all elements."/>
</category>
</seealso>
