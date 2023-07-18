# Quickstart

<!---IMPORT samples.docs.ArrayCreation-->
<!---IMPORT samples.docs.ArithmeticOperations-->
<!---IMPORT samples.docs.StandardOperations-->
<!---IMPORT samples.docs.IndexingAndIterating-->

<web-summary>
Start your journey with Multik using our Quickstart guide.
Grasp the basics and explore the power of multidimensional array operations through clear, concise examples.
</web-summary>

<card-summary>
Get to grips with Multik's core functionality through our hands-on Quickstart guide.
</card-summary>

<link-summary>
Learn Multik's essentials with our easy-to-follow Quickstart guide.
</link-summary>

## The Basics

In Multik, as in many similar libraries, the NDArray object is of great importance.
It represents homogeneous and single-typed numeric data with a multidimensional abstraction.
For multidimensionality, the following concepts are employed:

* **Dimension** — This mathematical concept refers to the number of coordinates required to determine a point in space.
  In our case, it refers to the necessary number of indexes for a data item.
  It is also often referred to as an `axis`.
* **Shape** — This is a set of the actual sizes for each `axis`.
* **Strides** — This is a set of the necessary number of steps to move to the next dimension in a given array.
  Steps are defined as indices;
  knowing what type of data the array stores,
  we can easily determine how many bytes are needed to move to the next axis.

These concepts help us operate on a simple array as if it were multidimensional.
Let's take a closer look at the ndarray itself and the properties that describe this array.

Take a simple matrix, which is actually a two-dimensional array:

```
[[1.5, 2.1, 3.0],
 [4.0, 5.0, 6.0]]
```

Let's assign this matrix to a variable arr. Then the following properties are available:

* **`arr.dim`** — Returns a dimension object, which characterizes the number of axes in the array.
  In this array, it will be `2D`.
  There are several such objects: `1D`, `2D`, `3D`, `4D`, and `ND`.
  Therefore, if we are working with arrays up to the fifth dimension,
  we can easily check the legitimacy of such operations at compile time.
  For larger dimensions, we don't have this capability.
    * **`arr.dim.d`** — Returns the number of axes in the array.

  > `1D`, `2D`, `3D`, and `4D` are subclasses of `ND`.
  > As a result, you can easily cast your matrix to `ND` and back.
  > We advise always specifying and operating with known dimensions, and only if your dimension is 5 or more, use `ND`.

* **`arr.shape`** — This is an array containing the sizes of the array for each axis.
  In our array, it will return `(2, 3)`, where `2` is the number of rows, and `3` is the number of columns in our
  matrix.
  The length of the shape is `dim.d`.
* **`arr.size`** — The number of elements in the array.
  You can obtain it by multiplying each number in shape.
  Specifically, `arr.size` will return `6`,
  and if you multiply each number in `arr.shape`, you will get the same result `(2 * 3)`.
* **`arr.dtype`** — The type of elements in the array.
  The supported types are: `Byte`, `Short`, `Int`, `Long`, `Float`, `Double`, `ComplexFloat`, and `ComplexDouble`.
  Because of the specific way arrays are stored, we can only operate with primitive types.
  > The `Boolean` type is not yet supported, and we will add it as soon as possible.
  >
  {style="note"}

### Array Creation

There are numerous ways to create an array.
A straightforward way to do so is by using the `mk[]` structure, which should be passed to the `mk.ndarray` method.
In this case, the array type will be inferred from the passed elements,
and the dimension will be determined based on the nesting of these data.
For instance:

<!---FUN simple_way_of_creation-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
println(a.dtype)
// DataType(nativeCode=3, itemSize=4, class=class kotlin.Int)
println(a.dim)
// dimension: 1
println(a)
// [1, 2, 3]

val b = mk.ndarray(mk[mk[1.5, 5.8], mk[9.1, 7.3]])
println(b.dtype)
// DataType(nativeCode=6, itemSize=8, class=class kotlin.Double)
println(b.dim)
// dimension: 2
println(b)
/*
 [[1.5, 5.8],
 [9.1, 7.3]]
 */
```

<!---END-->

You can create an array from Kotlin collections and standard arrays.

<!---FUN create_array_from_collections-->

```kotlin
mk.ndarray(setOf(1, 2, 3)) // [1, 2, 3]
listOf(8.4, 5.2, 9.3, 11.5).toNDArray() // [8.4, 5.2, 9.3, 11.5]
```

<!---END-->

Moreover, you can manually specify the size of each dimension.
For example, you can create a three-dimensional array from a regular array in this way.

<!---FUN create_array_from_primitive_with_shape-->

```kotlin
mk.ndarray(floatArrayOf(34.2f, 13.4f, 4.8f, 8.8f, 3.3f, 7.1f), 2, 1, 3)
/*
[[[34.2, 13.4, 4.8]],

[[8.8, 3.3, 7.1]]]
 */
```

<!---END-->

There are also standard functions that return an array filled with either zeros or ones,
namely the `zeros` and `ones` functions.
For these functions, you need to specify the element type.

<!---FUN create_zeros_and_ones_arrays-->

```kotlin
mk.zeros<Int>(7)
// [0, 0, 0, 0, 0, 0, 0]

mk.ones<Float>(3, 2)
/*
[[1.0, 1.0],
[1.0, 1.0],
[1.0, 1.0]]
 */
```

<!---END-->

In line with the Kotlin standard library, there are functions with a lambda.

<!---FUN creation_with_lambda-->

```kotlin
mk.d3array(2, 2, 3) { it * it } // create an array of dimension 3
/*
[[[0, 1, 4],
[9, 16, 25]],

[[36, 49, 64],
[81, 100, 121]]]
*/

mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, j) }
/*
[[0.0+(0.0)i, 0.0+(1.0)i, 0.0+(2.0)i],
[1.0+(0.0)i, 1.0+(1.0)i, 1.0+(2.0)i],
[2.0+(0.0)i, 2.0+(1.0)i, 2.0+(2.0)i]]
 */
```

<!---END-->

For numerical sequences, Multik provides two methods.
`arange` returns an array within a given range,
while `linspace` allows you to better control the number of numbers within a specified range.

<!---FUN creation_with_arange_and_linspace-->

```kotlin
mk.arange<Int>(3, 10, 2)
// [3, 5, 7, 9]
mk.linspace<Double>(0.0, 10.0, 8)
// [0.0, 1.4285714285714286, 2.857142857142857, 4.285714285714286, 5.714285714285714, 7.142857142857143, 8.571428571428571, 10.0]
```

<!---END-->

> For additional creation functions, please refer to the detailed API documentation.
>
> <a href="arange.md">arange</a>
> <a href="d1array.md">d1array</a>
> <a href="d2array.md">d2array</a>
> <a href="d2arrayIndices.md">d2arrayIndices</a>
> <a href="d3array.md">d3array</a>
> <a href="d3arrayIndices.md">d3arrayIndices</a>
> <a href="d4array.md">d4array</a>
> <a href="d4arrayIndices.md">d4arrayIndices</a>
> <a href="dnarray.md">dnarray</a>
> <a href="identity.md">identity</a>
> <a href="linspace.md">linspace</a>
> <a href="meshgrid.md">meshgrid</a>
> <a href="ndarray.md">ndarray</a>
> <a href="ndarrayOf.md">ndarrayOf</a>
> <a href="ones.md">ones</a>
> <a href="rand.md">rand</a>
> <a href="toNDArray.md">toNDArray</a>
> <a href="zeros.md">zeros</a>

### Arithmetic Operations

Arithmetic operations are performed element-wise on the array,
resulting in a new array filled with the outcome of the operation.

When operating on a scalar and an array, only the type must match; the shape of the array is retained.

<!---FUN arith_with_scalars-->

```kotlin
val a = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
println(3.3 + a)
/*
[[4.8, 5.4, 6.3],
[7.3, 8.3, 9.3]]
 */

println(a * 2.0)
/*
[[3.0, 4.2, 6.0],
[8.0, 10.0, 12.0]]
 */
```

<!---END-->

When conducting operations between two arrays, it's necessary that both the type,
dimensionality, and shape of the arrays match.
Dimensionality is checked at compile-time.
However, shape conformity can only be verified at runtime.
The operation remains element-wise, maintaining the original array shape.

<!---FUN div_with_ndarrays-->

```kotlin
val a = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
val b = mk.ndarray(mk[mk[1.0, 1.3, 3.0], mk[4.0, 9.5, 5.0]])
a / b // division
/*
[[1.5, 1.6153846153846154, 1.0],
[1.0, 0.5263157894736842, 1.2]]
*/
```

<!---END-->

Please note that the multiplication operator `*` performs element-wise operations.
For matrix product, use the [`dot`](dot.md) method.

<!---FUN mul_with_ndarrays-->

```kotlin
val a = mk.ndarray(mk[mk[0.5, 0.8, 0.0], mk[0.0, -4.5, 1.0]])
val b = mk.ndarray(mk[mk[1.0, 1.3, 3.0], mk[4.0, 9.5, 5.0]])
a * b // multiplication
/*
[[0.5, 1.04, 0.0],
[0.0, -42.75, 5.0]]
*/
```

<!---END-->

Operations such as `+=`, `-=`, `/=` and `*=` are designed to modify the current array directly,
without creating a new one, i.e., in-place.

<!---FUN inplace_arith_ops-->

```kotlin
val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
val b = mk.ndarray(mk[mk[4, 0], mk[7, 5]])

a += b
println(a)
/*
[[5, 2],
[10, 9]]
 */

a *= 3
println(a)
/*
[[15, 6],
[30, 27]]
 */
```

<!---END-->

### Basic Operations

Although Multik's NDArray does not implement
the [Collection](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/-collection/#kotlin.collections.Collection)
or [Iterable](https://kotlinlang.org/api/latest/jvm/stdlib/kotlin.collections/-iterable/) interfaces, it does offer a
subset of these methods for array operations.
Functions such as filter, map, reduce, among others, are available for use with NDArray objects.

<!---FUN small_example_collection_operations-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])
val b = a.filter { it > 2 }
println(b)  // [3, 4, 5]
val c = a.map { it * 2 }
println(c)  // [2, 4, 6, 8, 10]
val d = a.reduce { acc, value -> acc + value }
println(d)  // 15
```

<!---END-->

> See also
>
> <a href="all.md">all</a>
> <a href="any.md">any</a>
> <a href="asSequence.md">asSequence</a>
> <a href="associate.md">associate</a>
> <a href="associateBy.md">associateBy</a>
> <a href="associateByTo.md">associateByTo</a>
> <a href="associateTo.md">associateTo</a>
> <a href="associateWith.md">associateWith</a>
> <a href="associateWithTo.md">associateWithTo</a>
> <a href="average.md">average</a>
> <a href="chunked.md">chunked</a>
> <a href="contains.md">contains</a>
> <a href="count.md">count</a>
> <a href="distinct.md">distinct</a>
> <a href="distinctBy.md">distinctBy</a>
> <a href="drop.md">drop</a>
> <a href="dropWhile.md">dropWhile</a>
> <a href="filter.md">filter</a>
> <a href="filterIndexed.md">filterIndexed</a>
> <a href="filterMultiIndexed.md">filterMultiIndexed</a>
> <a href="filterNot.md">filterNot</a>
> <a href="find.md">find</a>
> <a href="findLast.md">findLast</a>
> <a href="first.md">first</a>
> <a href="firstOrNull.md">firstOrNull</a>
> <a href="flatMap.md">flatMap</a>
> <a href="flatMapIndexed.md">flatMapIndexed</a>
> <a href="flatMapMultiIndexed.md">flatMapMultiIndexed</a>
> <a href="flatMapMultiIndexed.md">flatMapMultiIndexed</a>
> <a href="fold.md">fold</a>
> <a href="foldIndexed.md">foldIndexed</a>
> <a href="foldMultiIndexed.md">foldMultiIndexed</a>
> <a href="forEach.md">forEach</a>
> <a href="forEachIndexed.md">forEachIndexed</a>
> <a href="forEachMultiIndexed.md">forEachMultiIndexed</a>
> <a href="groupNDArrayBy.md">groupNDArrayBy</a>
> <a href="groupNDArrayByTo.md">groupNDArrayByTo</a>
> <a href="indexOf.md">indexOf</a>
> <a href="indexOfFirst.md">indexOfFirst</a>
> <a href="indexOfLast.md">indexOfLast</a>
> <a href="intersect.md">intersect</a>
> <a href="joinTo.md">joinTo</a>
> <a href="joinToString.md">joinToString</a>
> <a href="last.md">last</a>
> <a href="lastIndexOf.md">lastIndexOf</a>
> <a href="lastOrNull.md">lastOrNull</a>
> <a href="map.md">map</a>
> <a href="minimum.md">minimum</a>
> <a href="maximum.md">maximum</a>
> <a href="maximum.md">maximum</a>
> <a href="mapIndexed.md">mapIndexed</a>
> <a href="mapMultiIndexed.md">mapMultiIndexed</a>
> <a href="mapIndexedNotNull.md">mapIndexedNotNull</a>
> <a href="mapNotNull.md">mapNotNull</a>
> <a href="max.md">max</a>
> <a href="maxBy.md">maxBy</a>
> <a href="maxWith.md">maxWith</a>
> <a href="min.md">min</a>
> <a href="minBy.md">minBy</a>
> <a href="minWith.md">minWith</a>
> <a href="onEach.md">onEach</a>
> <a href="partition.md">partition</a>
> <a href="windowed.md">windowed</a>
> <a href="reduce.md">reduce</a>
> <a href="reduceIndexed.md">reduceIndexed</a>
> <a href="reduceMultiIndexed.md">reduceMultiIndexed</a>
> <a href="reduceOrNull.md">reduceOrNull</a>
> <a href="reversed.md">reversed</a>
> <a href="scan.md">scan</a>
> <a href="scanIndexed.md">scanIndexed</a>
> <a href="scanMultiIndexed.md">scanMultiIndexed</a>
> <a href="sorted.md">sorted</a>
> <a href="sum.md">sum</a>
> <a href="sumBy.md">sumBy</a>
> <a href="toCollection.md">toCollection</a>
> <a href="toHashSet.md">toHashSet</a>
> <a href="toList.md">toList</a>
> <a href="toListD2.md">toListD2</a>
> <a href="toListD3.md">toListD3</a>
> <a href="toListD4.md">toListD4</a>
> <a href="toIntArray.md">toIntArray</a>
> <a href="toLongArray.md">toLongArray</a>
> <a href="toFloatArray.md">toFloatArray</a>
> <a href="toDoubleArray.md">toDoubleArray</a>
> <a href="toComplexFloatArray.md">toComplexFloatArray</a>
> <a href="toComplexDoubleArray.md">toComplexDoubleArray</a>
> <a href="toArray.md">toArray</a>
> <a href="toMutableList.md">toMutableList</a>
> <a href="toMutableSet.md">toMutableSet</a>
> <a href="toSet.md">toSet</a>
> <a href="toType.md">toType</a>

### Indexing, Slicing and Iterating

Multik provides intuitive ways to index, slice,
and iterate over NDArrays, similar to traditional collections with additional features for multidimensional arrays.

#### Indexing

In Multik, each index corresponds to a specific dimension (axis) of the array.
Here's how you can access elements in an NDArray:

<!---FUN simple_indexing-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
a[2] // select the element at index 2

val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
b[1, 2] // select the element at row 1 column 2
```

<!---END-->

#### Slicing

Multik introduces slicing, a feature that allows for creating sequences from arrays.
Slices are created by specifying the start, end, and step values within the array.

<!---FUN slice_1-->

```kotlin
val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
// select elements at rows 0 and 1 in column 1
b[0..<2, 1] // [2.1, 5.0]
```

<!---END-->

If all indices are not provided, the unmentioned ones are considered to be full slices — i.e., slices from start to end
with step 1 — retrieving all elements along the corresponding axis.

<!---FUN slice_2-->

```kotlin
val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
// select row 1
b[1] // [4.0, 5.0, 6.0]
b[1, 0..2..1] // [4.0, 5.0, 6.0]
```

<!---END-->

#### Iterating

Iterating over an NDArray in Multik is conducted element-wise, irrespective of the array's dimension:

<!---FUN iterating-->

```kotlin
val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
for (el in b) {
    print("$el, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0,
}
```

<!---END-->

To facilitate easy navigation through multidimensional arrays,
Multik provides multidimensional indices:

<!---FUN iterating_multiIndices-->

```kotlin
val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
for (index in b.multiIndices) {
    print("${b[index]}, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0,
}
```

<!---END-->

> See also
> 
> <a href="indexing-and-slicing.md">Indexing and Slicing</a>
> <a href="indexing-routines.md">Indexing routines</a>

## Shape Manipulation

[//]: # (TODO)

## Copies and Views

[//]: # (TODO)
