# sum

<web-summary>
API reference for sum() — returns the sum of all elements in an NDArray.
</web-summary>

<card-summary>
Compute the sum of all elements in an NDArray.
</card-summary>

<link-summary>
sum() — total sum of NDArray elements.
</link-summary>

Returns the sum of all elements in the array. `sumBy` variants allow mapping each element through
a selector function before summing, supporting `Int`, `Double`, `ComplexFloat`, and `ComplexDouble`
result types.

## Signatures

```kotlin
fun <T : Number, D : Dimension> MultiArray<T, D>.sum(): T

inline fun <T : Number, D : Dimension> MultiArray<T, D>.sumBy(
    selector: (T) -> Int
): Int

inline fun <T : Number, D : Dimension> MultiArray<T, D>.sumBy(
    selector: (T) -> Double
): Double
```

## Parameters

| Parameter  | Type            | Description                                     |
|------------|-----------------|-------------------------------------------------|
| `selector` | `(T) -> Int`    | Maps each element to an `Int` before summing.   |
| `selector` | `(T) -> Double` | Maps each element to a `Double` before summing. |

**Returns:** The sum of all elements (or transformed elements).

## Example

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])

a.sum()                      // 15
a.sumBy { it * it }          // 55 (sum of squares)

val b = mk.ndarray(mk[1.5, 2.5, 3.0])
b.sum()                      // 7.0
```

> `sumBy` also supports `ComplexFloat` and `ComplexDouble` selector types.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="average.md" summary="Arithmetic mean of elements."/>
  <a href="fold.md" summary="Custom accumulation."/>
  <a href="count.md" summary="Count elements."/>
  <a href="max.md" summary="Find the largest element."/>
</category>
</seealso>