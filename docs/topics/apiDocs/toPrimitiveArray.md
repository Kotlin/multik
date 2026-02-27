# toPrimitiveArray

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for toIntArray(), toFloatArray(), toDoubleArray(), and toLongArray() — convert
an NDArray to a Kotlin primitive array.
</web-summary>

<card-summary>
Convert an NDArray to a flat Kotlin primitive array (IntArray, DoubleArray, etc.).
</card-summary>

<link-summary>
toPrimitiveArray — convert NDArray to IntArray, DoubleArray, etc.
</link-summary>

Converts the array to a flat Kotlin primitive array (`IntArray`, `LongArray`, `FloatArray`,
`DoubleArray`, etc.) in row-major order. Multi-dimensional arrays are flattened.

## Signatures

```kotlin
fun <D : Dimension> MultiArray<Int, D>.toIntArray(): IntArray
fun <D : Dimension> MultiArray<Long, D>.toLongArray(): LongArray
fun <D : Dimension> MultiArray<Float, D>.toFloatArray(): FloatArray
fun <D : Dimension> MultiArray<Double, D>.toDoubleArray(): DoubleArray
fun <D : Dimension> MultiArray<ComplexFloat, D>.toComplexFloatArray(): ComplexFloatArray
fun <D : Dimension> MultiArray<ComplexDouble, D>.toComplexDoubleArray(): ComplexDoubleArray
```

**Returns:** A flat Kotlin primitive array containing all elements in row-major order.

## Example

<!---FUN toPrimitiveArray_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
a.toIntArray()       // IntArray [1, 2, 3]

val b = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
b.toDoubleArray()    // DoubleArray [1.0, 2.0, 3.0, 4.0]  (flat)
```

<!---END-->

> Multi-dimensional arrays are flattened to a 1D primitive array in **row-major** order.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="toArray.md" summary="Convert to nested Kotlin arrays."/>
  <a href="toList.md" summary="Convert to a List."/>
  <a href="toCollection.md" summary="Append to a mutable collection."/>
</category>
</seealso>
