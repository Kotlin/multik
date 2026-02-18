# Type

<!---IMPORT samples.docs.apiDocs.ArrayObjects-->

<web-summary>
Reference for the DataType enum — Multik's runtime type identification for NDArray elements.
Covers enum values, properties, helper methods, and type resolution.
</web-summary>

<card-summary>
DataType enum: runtime type identifiers, native codes, and type resolution helpers.
</card-summary>

<link-summary>
DataType enum — runtime element type identification for NDArrays.
</link-summary>

## DataType enum

`DataType` identifies the element type of an NDArray at runtime.
It is stored in the `dtype` property of every array.

```kotlin
public enum class DataType(
    public val nativeCode: Int,
    public val itemSize: Int,
    public val clazz: KClass<out Any>
)
```

### Values

| Constant                | nativeCode | itemSize | Kotlin class    |
|-------------------------|:----------:|:--------:|-----------------|
| `ByteDataType`          |     1      |    1     | `Byte`          |
| `ShortDataType`         |     2      |    2     | `Short`         |
| `IntDataType`           |     3      |    4     | `Int`           |
| `LongDataType`          |     4      |    8     | `Long`          |
| `FloatDataType`         |     5      |    4     | `Float`         |
| `DoubleDataType`        |     6      |    8     | `Double`        |
| `ComplexFloatDataType`  |     7      |    8     | `ComplexFloat`  |
| `ComplexDoubleDataType` |     8      |    16    | `ComplexDouble` |

### Properties

| Property     | Type              | Description                                     |
|--------------|-------------------|-------------------------------------------------|
| `nativeCode` | `Int`             | Integer identifier used by the JNI layer (1–8). |
| `itemSize`   | `Int`             | Bytes per element.                              |
| `clazz`      | `KClass<out Any>` | Corresponding Kotlin class.                     |

## Instance methods

### isNumber

```kotlin
fun isNumber(): Boolean
```

Returns `true` for types 1–6 (non-complex numeric types).

### isComplex

```kotlin
fun isComplex(): Boolean
```

Returns `true` for `ComplexFloatDataType` and `ComplexDoubleDataType`.

## Companion methods

### of (by native code)

```kotlin
fun of(i: Int): DataType
```

Returns the `DataType` matching the given `nativeCode`. Throws if `i` is out of range.

### of (by element)

```kotlin
fun <T : Any> of(element: T): DataType
```

Infers the `DataType` from an element instance.

### ofKClass

```kotlin
fun <T : Any> ofKClass(type: KClass<out T>): DataType
```

Resolves a `DataType` from a `KClass`.

## Usage

<!---FUN type_example-->

```kotlin
val a = mk.ndarray(mk[1.0, 2.0, 3.0])

a.dtype                          // DataType.DoubleDataType
a.dtype.nativeCode               // 6
a.dtype.itemSize                 // 8
a.dtype.isNumber()               // true
a.dtype.isComplex()              // false

DataType.of(3)                   // DataType.IntDataType
DataType.ofKClass(Float::class)  // DataType.FloatDataType
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="scalars.md" summary="Supported numeric and complex element types."/>
  <a href="ndarray-class.md" summary="NDArray class — properties, operators, and methods."/>
  <a href="dimension.md" summary="Compile-time dimension markers D1–D4 and DN."/>
</category>
</seealso>
