# toNDArray

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for the toNDArray() extension function — convert Kotlin
collections and primitive array matrices to Multik NDArrays.
</web-summary>

<card-summary>
Convert Iterables, nested Lists, and Array-of-arrays to NDArrays.
</card-summary>

<link-summary>
toNDArray() — convert Kotlin collections to NDArrays.
</link-summary>

## From Iterable (1D)

```kotlin
inline fun <reified T : Number> Iterable<T>.toNDArray(): D1Array<T>
inline fun <reified T : Complex> Iterable<T>.toNDArray(): D1Array<T>
```

<!---FUN toNDArray_iterable_example-->

```kotlin
val a = listOf(1, 2, 3).toNDArray()          // D1Array<Int>
val b = setOf(1.0, 2.0).toNDArray()          // D1Array<Double>
```

<!---END-->

## From nested Lists (2D–4D)

```kotlin
inline fun <reified T : Number> List<List<T>>.toNDArray(): D2Array<T>
inline fun <reified T : Complex> List<List<T>>.toNDArray(): D2Array<T>

inline fun <reified T : Number> List<List<List<T>>>.toNDArray(): D3Array<T>
inline fun <reified T : Complex> List<List<List<T>>>.toNDArray(): D3Array<T>

inline fun <reified T : Number> List<List<List<List<T>>>>.toNDArray(): D4Array<T>
inline fun <reified T : Complex> List<List<List<List<T>>>>.toNDArray(): D4Array<T>
```

<!---FUN toNDArray_nested_list_example-->

```kotlin
val m = listOf(listOf(1, 2), listOf(3, 4)).toNDArray()  // D2Array<Int>
```

<!---END-->

> All inner lists must have consistent sizes. Jagged lists throw an exception.
> {style="warning"}

## From Array of primitive arrays (2D)

```kotlin
fun Array<ByteArray>.toNDArray(): D2Array<Byte>
fun Array<ShortArray>.toNDArray(): D2Array<Short>
fun Array<IntArray>.toNDArray(): D2Array<Int>
fun Array<LongArray>.toNDArray(): D2Array<Long>
fun Array<FloatArray>.toNDArray(): D2Array<Float>
fun Array<DoubleArray>.toNDArray(): D2Array<Double>
```

<!---FUN toNDArray_array_of_arrays_example-->

```kotlin
val m = arrayOf(intArrayOf(1, 2), intArrayOf(3, 4)).toNDArray()  // D2Array<Int>
```

<!---END-->

> All inner arrays must have the same size.
> {style="warning"}

<seealso style="cards">
<category ref="api-docs">
  <a href="ndarray.md" summary="Create arrays from data with mk.ndarray()."/>
  <a href="ndarrayOf.md" summary="Create 1D arrays from varargs."/>
</category>
</seealso>
