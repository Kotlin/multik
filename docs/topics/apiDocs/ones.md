# ones

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.ones() — create NDArrays filled with ones.
</web-summary>

<card-summary>
Create arrays filled with ones of any shape and type.
</card-summary>

<link-summary>
mk.ones() — arrays filled with ones.
</link-summary>

## Signatures

```kotlin
inline fun <reified T : Any> Multik.ones(dim1: Int): D1Array<T>
inline fun <reified T : Any> Multik.ones(dim1: Int, dim2: Int): D2Array<T>
inline fun <reified T : Any> Multik.ones(dim1: Int, dim2: Int, dim3: Int): D3Array<T>
inline fun <reified T : Any> Multik.ones(dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<T>
inline fun <reified T : Any> Multik.ones(
    dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<T, DN>
```

### Generic with DataType

```kotlin
fun <T, D : Dimension> Multik.ones(dims: IntArray, dtype: DataType): NDArray<T, D>
```

## Parameters

| Parameter    | Type         | Description                                  |
|--------------|--------------|----------------------------------------------|
| `dim1..dim4` | `Int`        | Axis sizes. Must be positive.                |
| `dims`       | `vararg Int` | Additional axis sizes beyond the fourth.     |
| `dtype`      | `DataType`   | Element type (for the non-reified overload). |

**Returns:** NDArray of the given shape filled with ones.

## Example

<!---FUN ones_example-->

```kotlin
val v = mk.ones<Double>(5)           // [1.0, 1.0, 1.0, 1.0, 1.0]
val m = mk.ones<Int>(2, 3)           // [[1, 1, 1], [1, 1, 1]]
val t = mk.ones<Float>(2, 3, 4)     // shape (2, 3, 4), all 1.0f
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="zeros.md" summary="Create zero-filled arrays."/>
  <a href="identity.md" summary="Create identity matrices."/>
  <a href="ndarray.md" summary="Create arrays from data."/>
</category>
</seealso>