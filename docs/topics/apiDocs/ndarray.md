# ndarray

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.ndarray() — create NDArrays from nested lists, collections,
primitive arrays, and the mk[] DSL syntax.
</web-summary>

<card-summary>
Create NDArrays from lists, collections, primitive arrays, and the mk[] DSL.
</card-summary>

<link-summary>
mk.ndarray() — create arrays from lists, collections, and primitive arrays.
</link-summary>

## Overview

`mk.ndarray()` is the most common way to create arrays. It infers type and dimension from the input.
Use `mk[...]` to build nested list literals inline.

## From nested lists with mk[]

<!---FUN ndarray_from_mk_list_example-->

```kotlin
val v = mk.ndarray(mk[1, 2, 3])                         // D1Array<Int>
val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])     // D2Array<Double>
val t = mk.ndarray(mk[mk[mk[1, 2], mk[3, 4]]])         // D3Array<Int>
```

<!---END-->

Overloads exist for 1D–4D with both `Number` and `Complex` element types.

## From collection with shape

<!---FUN ndarray_from_collection_shape_example-->

```kotlin
val a = mk.ndarray<Int, D2>(listOf(1, 2, 3, 4), intArrayOf(2, 2))
// D2Array<Int>, shape (2, 2)
```

<!---END-->

| Parameter  | Type            | Description                                       |
|------------|-----------------|---------------------------------------------------|
| `elements` | `Collection<T>` | Flat collection of values.                        |
| `shape`    | `IntArray`      | Target shape. Product must equal `elements.size`. |

Dimension can be reified or passed explicitly:

<!---FUN ndarray_from_collection_explicit_dim_example-->

```kotlin
val b = mk.ndarray(listOf(1, 2, 3, 4), intArrayOf(2, 2), D2)
```

<!---END-->

## From collection with dimension sizes

<!---FUN ndarray_from_collection_dims_example-->

```kotlin
val m = mk.ndarray(listOf(1, 2, 3, 4, 5, 6), 2, 3)     // D2Array, shape (2, 3)
val t = mk.ndarray(listOf(1, 2, 3, 4, 5, 6), 1, 2, 3)  // D3Array, shape (1, 2, 3)
```

<!---END-->

Overloads for 2D, 3D, 4D, and N-D (5+ dimensions via varargs).

## From primitive arrays

1D:

<!---FUN ndarray_from_primitive_1d_example-->

```kotlin
val a = mk.ndarray(intArrayOf(1, 2, 3))         // D1Array<Int>
val b = mk.ndarray(doubleArrayOf(1.0, 2.0))     // D1Array<Double>
```

<!---END-->

With shape:

<!---FUN ndarray_from_primitive_shape_example-->

```kotlin
val m = mk.ndarray(intArrayOf(1, 2, 3, 4), 2, 2)  // D2Array<Int>
```

<!---END-->

Supported types: `ByteArray`, `ShortArray`, `IntArray`, `LongArray`, `FloatArray`, `DoubleArray`, `ComplexFloatArray`,
`ComplexDoubleArray`.

## From Array of primitive arrays

<!---FUN ndarray_from_array_of_arrays_example-->

```kotlin
val m = mk.ndarray(arrayOf(intArrayOf(1, 2), intArrayOf(3, 4)))  // D2Array<Int>
```

<!---END-->

> All inner arrays must have the same size. An exception is thrown for jagged arrays.
> {style="warning"}

## Pitfalls

* The product of the shape must exactly equal the number of elements. A mismatch throws `IllegalArgumentException`.
* Nested list inputs are validated for consistent sizes — jagged lists are rejected.

<seealso style="cards">
<category ref="api-docs">
  <a href="ndarrayOf.md" summary="Create 1D arrays from varargs."/>
  <a href="toNDArray.md" summary="Convert Kotlin collections to NDArrays."/>
  <a href="d1array.md" summary="Create 1D arrays with an init lambda."/>
</category>
</seealso>
