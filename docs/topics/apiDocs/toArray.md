# toArray

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for toArray() — converts a 2D, 3D, or 4D NDArray into nested Kotlin arrays.
</web-summary>

<card-summary>
Convert a multi-dimensional NDArray to nested Kotlin arrays.
</card-summary>

<link-summary>
toArray() — convert NDArray to nested Kotlin arrays.
</link-summary>

Converts a 2D, 3D, or 4D NDArray into nested Kotlin arrays that mirror the dimensional structure.
Available for all numeric and complex element types.

## Signatures

```kotlin
// 2D
fun MultiArray<Int, D2>.toArray(): Array<IntArray>
fun MultiArray<Long, D2>.toArray(): Array<LongArray>
fun MultiArray<Float, D2>.toArray(): Array<FloatArray>
fun MultiArray<Double, D2>.toArray(): Array<DoubleArray>
fun MultiArray<ComplexFloat, D2>.toArray(): Array<ComplexFloatArray>
fun MultiArray<ComplexDouble, D2>.toArray(): Array<ComplexDoubleArray>

// 3D
fun MultiArray<Int, D3>.toArray(): Array<Array<IntArray>>
fun MultiArray<Double, D3>.toArray(): Array<Array<DoubleArray>>
// (also for Long, Float, ComplexFloat, ComplexDouble)

// 4D
fun MultiArray<Int, D4>.toArray(): Array<Array<Array<IntArray>>>
fun MultiArray<Double, D4>.toArray(): Array<Array<Array<DoubleArray>>>
// (also for Long, Float, ComplexFloat, ComplexDouble)
```

**Returns:** Nested Kotlin arrays matching the dimensional structure.

## Example

<!---FUN toArray_example-->

```kotlin
val m = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

val arr: Array<IntArray> = m.toArray()
// arr[0] = IntArray [1, 2, 3]
// arr[1] = IntArray [4, 5, 6]
```

<!---END-->

> `toArray()` is available for D2, D3, and D4 arrays only. For 1D arrays, use
> [`toPrimitiveArray`](toPrimitiveArray.md) functions instead.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="toPrimitiveArray.md" summary="Convert to a flat primitive array."/>
  <a href="toList.md" summary="Convert to a List or nested lists."/>
  <a href="toCollection.md" summary="Append to a mutable collection."/>
</category>
</seealso>
