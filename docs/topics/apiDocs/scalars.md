# Scalars

<!---IMPORT samples.docs.apiDocs.ArrayObjects-->

<web-summary>
Reference for scalar element types supported by Multik NDArrays:
Byte, Short, Int, Long, Float, Double, ComplexFloat, and ComplexDouble.
</web-summary>

<card-summary>
Supported NDArray element types — numeric primitives and complex numbers.
</card-summary>

<link-summary>
Scalar types: Byte, Short, Int, Long, Float, Double, ComplexFloat, ComplexDouble.
</link-summary>

## Supported types

Multik arrays store elements of a single scalar type.
All scalars implement Kotlin's `Number` or the `Complex` interface.

| Type            | Size (bytes) | Range / Precision           | DataType constant       |
|-----------------|:------------:|-----------------------------|-------------------------|
| `Byte`          |      1       | -128 to 127                 | `ByteDataType`          |
| `Short`         |      2       | -32 768 to 32 767           | `ShortDataType`         |
| `Int`           |      4       | -2^31 to 2^31-1             | `IntDataType`           |
| `Long`          |      8       | -2^63 to 2^63-1             | `LongDataType`          |
| `Float`         |      4       | ~7 decimal digits           | `FloatDataType`         |
| `Double`        |      8       | ~15 decimal digits          | `DoubleDataType`        |
| `ComplexFloat`  |      8       | Two `Float` parts (re, im)  | `ComplexFloatDataType`  |
| `ComplexDouble` |      16      | Two `Double` parts (re, im) | `ComplexDoubleDataType` |

## Numeric types

The first six types map directly to Kotlin primitives.
Arithmetic operators (`+`, `-`, `*`, `/`) and in-place variants (`+=`, `-=`, `*=`, `/=`) are supported element-wise on
arrays of these types.

```kotlin
val a = mk.ndarray(mk[1, 2, 3])       // Int
val b = mk.ndarray(mk[1.0, 2.0, 3.0]) // Double
```

## Complex types

### ComplexFloat

```kotlin
@JvmInline
public value class ComplexFloat(private val number: Long) : Complex
```

| Property / Method               | Type           | Description                              |
|---------------------------------|----------------|------------------------------------------|
| `re`                            | `Float`        | Real part.                               |
| `im`                            | `Float`        | Imaginary part.                          |
| `conjugate()`                   | `ComplexFloat` | Returns `a - bi`.                        |
| `abs()`                         | `Float`        | Magnitude `sqrt(re² + im²)`.             |
| `angle()`                       | `Float`        | Phase angle in radians.                  |
| `component1()` / `component2()` | `Float`        | Destructuring support: `val (r, i) = c`. |

**Constants:** `ComplexFloat.zero`, `ComplexFloat.one`, `ComplexFloat.NaN`

### ComplexDouble

```kotlin
public interface ComplexDouble : Complex
```

Same API as `ComplexFloat` but with `Double` precision.

**Constants:** `ComplexDouble.zero`, `ComplexDouble.one`, `ComplexDouble.NaN`

### Creating complex values

<!---FUN creating_complex_values_example-->

```kotlin
val c1 = ComplexFloat(1.0f, 2.0f)   // 1.0 + 2.0i
val c2 = ComplexDouble(3.0, -1.0)   // 3.0 - 1.0i

// Factory helpers on Complex companion
val real = Complex.r(5.0)           // 5.0 + 0.0i
val imag = Complex.i(3.0)           // 0.0 + 3.0i
```

<!---END-->

### ComplexFloatArray / ComplexDoubleArray

Backing storage for complex arrays. Each element occupies two primitive slots internally.

<!---FUN creating_complex_array_example-->

```kotlin
val arr = ComplexFloatArray(3) { ComplexFloat(it.toFloat(), 0f) }
arr[0] // ComplexFloat(0.0, 0.0)
```

<!---END-->

## Type conversion

Use `asType<T>()` or `toType<T>()` to convert between scalar types.
Converting a numeric type to a complex type sets the imaginary part to zero.
See [](type.md) for `DataType` details and [](ndarray-class.md) for conversion methods.

> Narrowing conversions (e.g. `Double` → `Float`, `Long` → `Int`) may lose precision or truncate. This is standard
> Kotlin `Number` conversion behavior.
> {style="warning"}

<seealso style="cards">
<category ref="api-docs">
  <a href="type.md" summary="DataType enum for runtime type identification."/>
  <a href="ndarray-class.md" summary="NDArray class — properties, operators, and methods."/>
</category>
</seealso>
