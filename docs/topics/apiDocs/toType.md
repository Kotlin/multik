# toType

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for toType() — converts NDArray element type to a different numeric type
(e.g. Int to Double).
</web-summary>

<card-summary>
Convert NDArray element type — e.g. Int to Double, Float to Long.
</card-summary>

<link-summary>
toType() — element type conversion for NDArrays.
</link-summary>

## Signatures

```kotlin
inline fun <T, reified O : Any, D : Dimension> MultiArray<T, D>.toType(
    copy: CopyStrategy = CopyStrategy.FULL
): NDArray<O, D>

fun <T, O : Any, D : Dimension> MultiArray<T, D>.toType(
    dtype: DataType,
    copy: CopyStrategy = CopyStrategy.FULL
): NDArray<O, D>
```

## Parameters

| Parameter | Type           | Description                                                           |
|-----------|----------------|-----------------------------------------------------------------------|
| `O`       | (reified type) | Target element type.                                                  |
| `dtype`   | `DataType`     | Target type as a `DataType` enum value.                               |
| `copy`    | `CopyStrategy` | Copy behavior. `FULL` (default) always copies; other strategies may share data when types match. |

**Returns:** `NDArray<O, D>` — a new array with the same shape and converted element type.

## Example

<!---FUN toType_example-->

```kotlin
val ints = mk.ndarray(mk[1, 2, 3])

val doubles = ints.toType<Int, Double, D1>()
// D1Array<Double> [1.0, 2.0, 3.0]

val floats = ints.toType<Int, Float, D1>()
// D1Array<Float> [1.0, 2.0, 3.0]
```

<!---END-->

## Pitfalls

* Narrowing conversions (e.g. `Double` to `Int`) truncate values — `2.9` becomes `2`.
* When source and target types match with `CopyStrategy.FULL`, a copy is still made.

<seealso style="cards">
<category ref="api-docs">
  <a href="toPrimitiveArray.md" summary="Convert to a Kotlin primitive array."/>
  <a href="map.md" summary="Transform elements with a custom function."/>
  <a href="type.md" summary="NDArray type system reference."/>
</category>
</seealso>