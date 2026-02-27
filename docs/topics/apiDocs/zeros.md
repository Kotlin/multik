# zeros

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.zeros() — create NDArrays filled with zeros.
</web-summary>

<card-summary>
Create arrays filled with zeros of any shape and type.
</card-summary>

<link-summary>
mk.zeros() — arrays filled with zeros.
</link-summary>

Creates an array of the given shape filled with zeros. Overloads are available for 1D through 4D,
N-dimensional (5+), and generic shape-based creation.

## Signatures

```kotlin
inline fun <reified T : Any> Multik.zeros(dim1: Int): D1Array<T>
inline fun <reified T : Any> Multik.zeros(dim1: Int, dim2: Int): D2Array<T>
inline fun <reified T : Any> Multik.zeros(dim1: Int, dim2: Int, dim3: Int): D3Array<T>
inline fun <reified T : Any> Multik.zeros(dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<T>
inline fun <reified T : Any> Multik.zeros(
    dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int
): NDArray<T, DN>
```

### Generic with DataType

```kotlin
fun <T, D : Dimension> Multik.zeros(dims: IntArray, dtype: DataType): NDArray<T, D>
```

## Parameters

| Parameter    | Type         | Description                                  |
|--------------|--------------|----------------------------------------------|
| `dim1..dim4` | `Int`        | Axis sizes. Must be positive.                |
| `dims`       | `vararg Int` | Additional axis sizes beyond the fourth.     |
| `dtype`      | `DataType`   | Element type (for the non-reified overload). |

**Returns:** NDArray of the given shape filled with zeros.

## Example

<!---FUN zeros_example-->

```kotlin
val v = mk.zeros<Double>(5)           // [0.0, 0.0, 0.0, 0.0, 0.0]
val m = mk.zeros<Int>(2, 3)           // [[0, 0, 0], [0, 0, 0]]
val t = mk.zeros<Float>(2, 3, 4)     // shape (2, 3, 4), all 0.0f
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="ones.md" summary="Create one-filled arrays."/>
  <a href="identity.md" summary="Create identity matrices."/>
  <a href="ndarray.md" summary="Create arrays from data."/>
</category>
</seealso>
