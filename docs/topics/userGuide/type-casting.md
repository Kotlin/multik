# Type casting

<!---IMPORT samples.docs.userGuide.TypeCasting-->

<web-summary>
Learn how to change ndarray element types in Multik using `asType` and `toType`.
Understand the difference between full and meaningful copies when converting types.
</web-summary>

<card-summary>
Convert ndarrays between numeric and complex types with `asType` and `toType`.
Choose copy strategies to control memory and views.
</card-summary>

<link-summary>
Cast ndarrays between numeric types and control copying with `toType`.
</link-summary>

## Overview

Type casting changes the dtype of an ndarray while preserving its shape and dimension.
Multik provides two primary APIs:

* `asType<T>()` on `NDArray`
* `toType<T>()` on `MultiArray`

Both create a new array with converted element values.

## `asType` on NDArray

`asType` is a simple way to cast an `NDArray` to a new numeric type.
It preserves shape and dimension.

<!---FUN asType_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
val b = a.asType<Double>()
// dtype is Double, values are [1.0, 2.0, 3.0]
```

<!---END-->

You can also cast using a `DataType`:

<!---FUN dataType_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
val c = a.asType<Float>(DataType.FloatDataType)
```

<!---END-->

## `toType` on MultiArray

`toType` works for any `MultiArray` and supports copy strategies.
It can be useful when you are working with views or with arrays that are not consistent.

<!---FUN toType_example-->

```kotlin
val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
val d = m.toType<Int, Double, D2>()
```

<!---END-->

### Copy strategies

`toType` accepts a `CopyStrategy`:

* `CopyStrategy.FULL` (default) copies the entire underlying storage.
* `CopyStrategy.MEANINGFUL` copies only the visible elements of the array.

<!---FUN view_copy_example-->

```kotlin
val view = m[0] // view of the first row

val fullCopy = view.toType<Int, Double, D1>(CopyStrategy.FULL)
val meaningfulCopy = view.toType<Int, Double, D1>(CopyStrategy.MEANINGFUL)
```

<!---END-->

Use `MEANINGFUL` when you want to materialize just the data visible through a view.
Use `FULL` when you want to preserve the full underlying storage layout.

## Casting to complex types

Numeric arrays can be converted to complex types; the imaginary part is set to zero:

<!---FUN casting_to_complex_example-->

```kotlin
val real = mk.ndarray(mk[1.0, 2.0])
val complex = real.toType<Double, ComplexDouble, D1>()
```

<!---END-->

Conversions between complex types are supported as well.

## Notes and tips

* Casting always creates a new array; it does not modify the original.
* When the dtype is already the target type, `toType` returns a copy based on the chosen strategy.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="creating-multidimensional-arrays.md" summary="Build ndarrays from literals, collections, primitives, and factories."/>
  <a href="shape-manipulation.md" summary="Reshape, transpose, squeeze, and combine arrays."/>
  <a href="input-and-output.md" summary="Convert arrays and use file I/O on the JVM."/>
</category>
</seealso>
