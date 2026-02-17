# NDArray

<web-summary>
Complete API reference for the NDArray class — Multik's core multidimensional array.
Covers the constructor, properties, indexing operators, shape manipulation,
type conversion, copy semantics, and concatenation.
</web-summary>

<card-summary>
NDArray class reference: constructor, properties, operators, reshape, transpose, copy, and concatenation.
</card-summary>

<link-summary>
NDArray class API — properties, operators, reshape, transpose, copy, and more.
</link-summary>

## Declaration

```kotlin
public class NDArray<T, D : Dimension>(
    data: ImmutableMemoryView<T>,
    offset: Int = 0,
    shape: IntArray,
    strides: IntArray = computeStrides(shape),
    dim: D,
    base: MultiArray<T, out Dimension>? = null
) : MutableMultiArray<T, D>
```

**Type parameters:**

| Parameter | Description                                                                                        |
|-----------|----------------------------------------------------------------------------------------------------|
| `T`       | Element type (`Byte`, `Short`, `Int`, `Long`, `Float`, `Double`, `ComplexFloat`, `ComplexDouble`). |
| `D`       | [Dimension](dimension.md) marker (`D1`, `D2`, `D3`, `D4`, or `DN`).                                |

**Inheritance:** `NDArray` → `MutableMultiArray` → `MultiArray`

## Type aliases

| Alias        | Expands to       |
|--------------|------------------|
| `D1Array<T>` | `NDArray<T, D1>` |
| `D2Array<T>` | `NDArray<T, D2>` |
| `D3Array<T>` | `NDArray<T, D3>` |
| `D4Array<T>` | `NDArray<T, D4>` |

## Properties

| Property       | Type                            | Description                                                                           |
|----------------|---------------------------------|---------------------------------------------------------------------------------------|
| `data`         | `MemoryView<T>`                 | Underlying flat storage.                                                              |
| `offset`       | `Int`                           | Start index in `data`.                                                                |
| `shape`        | `IntArray`                      | Size of each axis. For a 2×3 matrix: `intArrayOf(2, 3)`.                              |
| `strides`      | `IntArray`                      | Element stride per axis. Computed from `shape` by default.                            |
| `size`         | `Int`                           | Total element count (product of `shape`).                                             |
| `dim`          | `D`                             | Dimension instance (`D1`, `D2`, …).                                                   |
| `dtype`        | `DataType`                      | Element [data type](type.md).                                                         |
| `base`         | `MultiArray<T, out Dimension>?` | Source array when this is a view; `null` for owned arrays.                            |
| `consistent`   | `Boolean`                       | `true` when `offset == 0`, `size == data.size`, and strides match the default layout. |
| `indices`      | `IntRange`                      | `0 until size` — valid flat indices for 1D access.                                    |
| `multiIndices` | `MultiIndexProgression`         | Range of all valid multi-dimensional index tuples.                                    |

## Indexing operators

### Get

```kotlin
// 1D
operator fun get(index: Int): T

// 2D
operator fun get(ind1: Int, ind2: Int): T

// 3D
operator fun get(ind1: Int, ind2: Int, ind3: Int): T

// 4D
operator fun get(ind1: Int, ind2: Int, ind3: Int, ind4: Int): T
```

Index values are zero-based. Negative indices are **not** supported — use `shape[axis] - k` instead.

### Set

```kotlin
operator fun set(index: Int, value: T)
operator fun set(ind1: Int, ind2: Int, value: T)
operator fun set(ind1: Int, ind2: Int, ind3: Int, value: T)
operator fun set(ind1: Int, ind2: Int, ind3: Int, ind4: Int, value: T)
```

### Slice access

Ranges and `Slice` objects can be used in place of `Int` indices.
See [Indexing routines](indexing-routines.md) for the full slicing API.

```kotlin
val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])
a[0..1, 1..2] // 2D subarray
```

## Shape manipulation

### reshape

Returns a view with a different shape but the same total size.

```kotlin
fun reshape(dim1: Int): D1Array<T>
fun reshape(dim1: Int, dim2: Int): D2Array<T>
fun reshape(dim1: Int, dim2: Int, dim3: Int): D3Array<T>
fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): D4Array<T>
fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int): NDArray<T, DN>
```

Throws `IllegalArgumentException` if the product of the new shape does not equal `size`.

### transpose

Reverses or permutes axes.

```kotlin
fun transpose(vararg axes: Int = intArrayOf()): NDArray<T, D>
```

With no arguments, reverses all axes. With arguments, reorders axes to the given permutation.

### squeeze / unsqueeze

```kotlin
fun squeeze(vararg axes: Int): NDArray<T, DN>   // remove size-1 axes
fun unsqueeze(vararg axes: Int): NDArray<T, DN> // insert size-1 axes
```

### flatten

```kotlin
fun flatten(): D1Array<T>
```

Returns a 1D copy. If the array is already consistent, shares the underlying data.

## Type conversion

### asType

```kotlin
inline fun <reified E : Number> asType(): NDArray<E, D>
fun <E : Number> asType(dataType: DataType): NDArray<E, D>
```

Creates a new array with elements converted to `E`. Shape and dimension are preserved.

### Dimension casts

```kotlin
fun asD1Array(): D1Array<T>
fun asD2Array(): D2Array<T>
fun asD3Array(): D3Array<T>
fun asD4Array(): D4Array<T>
fun asDNArray(): NDArray<T, DN>
```

Cast the dimension type parameter. Throws `ClassCastException` if the actual dimension does not match.

## Copy operations

| Method       | Behavior                                                                                       |
|--------------|------------------------------------------------------------------------------------------------|
| `copy()`     | Shallow copy — new `NDArray` with the same `data` reference but independent `shape`/`strides`. |
| `deepCopy()` | Full copy — allocates new storage and copies all elements.                                     |

> Slicing returns a view that shares data with the original. Use `deepCopy()` when you need an independent array.
> {style="note"}

## Concatenation

```kotlin
infix fun cat(other: MultiArray<T, D>): NDArray<T, D>             // axis 0
fun cat(other: MultiArray<T, D>, axis: Int = 0): NDArray<T, D>    // given axis
fun cat(other: List<MultiArray<T, D>>, axis: Int = 0): NDArray<T, D>
```

All operands must have the same shape on every axis except the concatenation axis.

## Utility

| Method          | Returns       | Description                                                          |
|-----------------|---------------|----------------------------------------------------------------------|
| `isScalar()`    | `Boolean`     | `true` if the array holds a single element or has an empty shape.    |
| `isEmpty()`     | `Boolean`     | `true` if `size == 0`.                                               |
| `isNotEmpty()`  | `Boolean`     | `true` if `size > 0`.                                                |
| `iterator()`    | `Iterator<T>` | Flat element iterator respecting `offset` and `strides`.             |
| `equals(other)` | `Boolean`     | Element-wise equality (handles `ComplexFloat`/`ComplexDouble`).      |
| `hashCode()`    | `Int`         | Hash based on all elements.                                          |
| `toString()`    | `String`      | Pretty-printed representation up to 4D; truncated for larger arrays. |

## See also

<seealso style="cards">
<category ref="api-docs">
  <a href="scalars.md" summary="Supported numeric and complex element types."/>
  <a href="type.md" summary="DataType enum for runtime type identification."/>
  <a href="dimension.md" summary="Compile-time dimension markers D1–D4 and DN."/>
  <a href="indexing-routines.md" summary="Slice, RInt, and view helpers."/>
</category>
</seealso>