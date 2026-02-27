# dnarray

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.dnarray() — create an N-dimensional NDArray with a flat-index init lambda.
</web-summary>

<card-summary>
Create an N-dimensional array where each element is computed by a flat-index init function.
</card-summary>

<link-summary>
mk.dnarray() — N-D array from flat-index init lambda.
</link-summary>

Creates an N-dimensional array where each element is computed by a flat-index `init` function.
Use the explicit-dimensions overload for 5+ dimensions, or the shape-based overload to create
arrays of any dimension from a dynamically computed shape.

## Signatures

### From explicit dimensions (5+)

```kotlin
inline fun <reified T : Any> Multik.dnarray(
    sizeD1: Int,
    sizeD2: Int,
    sizeD3: Int,
    sizeD4: Int,
    vararg dims: Int,
    noinline init: (Int) -> T
): NDArray<T, DN>
```

### From shape array

```kotlin
inline fun <reified T : Any, reified D : Dimension> Multik.dnarray(
    shape: IntArray,
    noinline init: (Int) -> T
): NDArray<T, D>
```

## Parameters

| Parameter        | Type         | Description                                                                |
|------------------|--------------|----------------------------------------------------------------------------|
| `sizeD1..sizeD4` | `Int`        | First four axis sizes. Must be positive.                                   |
| `dims`           | `vararg Int` | Additional axis sizes beyond the fourth.                                   |
| `shape`          | `IntArray`   | Complete shape array. All elements must be positive.                       |
| `init`           | `(Int) -> T` | Lambda that receives a flat index. Elements are filled in row-major order. |

**Returns:** `NDArray<T, DN>` (varargs version) or `NDArray<T, D>` (shape version)

## Example

<!---FUN dnarray_example-->

```kotlin
// 5D array
val a = mk.dnarray<Int>(2, 3, 4, 5, 6) { it }
// shape (2, 3, 4, 5, 6), 720 elements

// Using shape array with reified dimension
val b = mk.dnarray<Double, D3>(intArrayOf(2, 3, 4)) { it.toDouble() }
// shape (2, 3, 4)
```

<!---END-->

> The shape-based overload lets you create arrays of any dimension (including D1–D4) from a dynamically computed shape.
> {style="tip"}

<seealso style="cards">
<category ref="api-docs">
  <a href="d4array.md" summary="Create 4D arrays with a flat-index init lambda."/>
  <a href="ndarray.md" summary="Create arrays from lists and collections."/>
  <a href="zeros.md" summary="Create zero-filled arrays of any shape."/>
</category>
</seealso>
