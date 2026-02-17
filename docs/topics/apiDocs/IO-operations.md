# IO operations

<web-summary>
API reference for Multik I/O: JVM-only file operations for CSV, NPY, and NPZ formats,
plus cross-platform in-memory conversions to lists, sets, and primitive arrays.
</web-summary>

<card-summary>
File I/O (CSV, NPY, NPZ) on the JVM and cross-platform in-memory conversions.
</card-summary>

<link-summary>
I/O API — CSV, NPY, NPZ file operations and in-memory array conversions.
</link-summary>

## Platform support

| Operation                | JVM | JS  | WASM | Native |
|--------------------------|:---:|:---:|:----:|:------:|
| File I/O (CSV, NPY, NPZ) | Yes |  —  |  —   |   —    |
| In-memory conversions    | Yes | Yes | Yes  |  Yes   |

## File I/O (JVM only) {id="file-io"}

All file I/O functions are extension functions on the `Multik` (`mk`) object.
Import from `org.jetbrains.kotlinx.multik.api.io`.

### Generic read / write

```kotlin
fun <T, D : Dimension> Multik.read(path: Path): NDArray<T, D>
fun Multik.write(path: Path, ndarray: NDArray<*, *>)
```

The format is inferred from the file extension (`.npy` or `.csv`). NPZ must be read with `readNPZ` explicitly.

### NPY (NumPy binary format) {id="npy"}

```kotlin
fun <T, D : Dimension> Multik.readNPY(path: String): NDArray<T, D>
fun <T, D : Dimension> Multik.readNPY(path: File): NDArray<T, D>
fun <T, D : Dimension> Multik.readNPY(path: Path): NDArray<T, D>

fun <T, D : Dimension> Multik.writeNPY(path: String, ndarray: NDArray<T, D>)
fun <T, D : Dimension> Multik.writeNPY(path: File, ndarray: NDArray<T, D>)
fun <T, D : Dimension> Multik.writeNPY(path: Path, ndarray: NDArray<T, D>)
```

* **Supported types:** `Byte`, `Short`, `Int`, `Long`, `Float`, `Double` (no complex).
* **Dimensions:** any.
* Binary format — compact and preserves exact dtype and shape.

```kotlin
val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
mk.writeNPY("matrix.npy", a)

val loaded = mk.readNPY<Double, D2>("matrix.npy")
```

### NPZ (NumPy compressed archive) {id="npz"}

```kotlin
fun Multik.readNPZ(path: String): List<NDArray<out Number, out DimN>>
fun Multik.readNPZ(path: File): List<NDArray<out Number, out DimN>>
fun Multik.readNPZ(path: Path): List<NDArray<out Number, out DimN>>

fun Multik.writeNPZ(path: Path, vararg ndArrays: NDArray<*, *>)
fun Multik.writeNPZ(path: Path, ndArrays: List<NDArray<*, *>>)
```

* **Supported types:** same as NPY (numeric only).
* Arrays in the archive are named `arr_0`, `arr_1`, etc.
* Wrapper around ZIP compression.

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
val b = mk.ndarray(mk[4.0, 5.0])

mk.writeNPZ(Path.of("bundle.npz"), a, b)
val arrays = mk.readNPZ("bundle.npz") // List of 2 arrays
```

### CSV (comma-separated values) {id="csv"}

```kotlin
fun <T, D : Dimension> Multik.readCSV(
    fileName: String, delimiter: Char = ',', charset: Charset = Charsets.UTF_8
): NDArray<T, D>

fun <T, D : Dimension> Multik.readCSV(
    file: File, delimiter: Char = ',', charset: Charset = Charsets.UTF_8
): NDArray<T, D>

fun <T, D : Dimension> Multik.writeCSV(
    path: String, ndarray: NDArray<T, D>, delimiter: Char = ','
)

fun <T, D : Dimension> Multik.writeCSV(
    file: File, ndarray: NDArray<T, D>, delimiter: Char = ','
)
```

* **Supported types:** all 8 types including `ComplexFloat` and `ComplexDouble`.
* **Dimensions:** D1 and D2 only.
* Supports `.gz` and `.zip` compressed input files.
* Complex values are stored as two space-separated numbers (e.g. `"1.5 2.3"`).
* Delimiter is configurable — use `'\t'` for TSV, `';'` for semicolons, etc.

```kotlin
val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
mk.writeCSV("data.csv", m)

val loaded = mk.readCSV<Double, D2>("data.csv")
```

## In-memory conversions (all platforms) {id="in-memory"}

### To collections

```kotlin
fun <T> MultiArray<T, *>.toList(): List<T>
fun <T> MultiArray<T, *>.toMutableList(): MutableList<T>
fun <T> MultiArray<T, *>.toSet(): Set<T>
fun <T> MultiArray<T, *>.toMutableSet(): MutableSet<T>
```

### To nested lists

```kotlin
fun <T> MultiArray<T, D2>.toListD2(): List<List<T>>
fun <T> MultiArray<T, D3>.toListD3(): List<List<List<T>>>
fun <T> MultiArray<T, D4>.toListD4(): List<List<List<List<T>>>>
```

### To primitive arrays (1D)

```kotlin
fun MultiArray<Byte, D1>.toByteArray(): ByteArray
fun MultiArray<Short, D1>.toShortArray(): ShortArray
fun MultiArray<Int, D1>.toIntArray(): IntArray
fun MultiArray<Long, D1>.toLongArray(): LongArray
fun MultiArray<Float, D1>.toFloatArray(): FloatArray
fun MultiArray<Double, D1>.toDoubleArray(): DoubleArray
fun MultiArray<ComplexFloat, D1>.toComplexFloatArray(): ComplexFloatArray
fun MultiArray<ComplexDouble, D1>.toComplexDoubleArray(): ComplexDoubleArray
```

### To nested primitive arrays (2D–4D)

```kotlin
fun <T> MultiArray<T, D2>.toArray(): Array<*>   // e.g. Array<DoubleArray>
fun <T> MultiArray<T, D3>.toArray(): Array<*>   // e.g. Array<Array<DoubleArray>>
fun <T> MultiArray<T, D4>.toArray(): Array<*>   // e.g. Array<Array<Array<DoubleArray>>>
```

The inner array type matches the element dtype.

## Format comparison

|                      | NPY          | NPZ          | CSV                 |
|----------------------|--------------|--------------|---------------------|
| **Binary**           | Yes          | Yes (ZIP)    | No (text)           |
| **Types**            | Numeric only | Numeric only | All (incl. complex) |
| **Dimensions**       | Any          | Any          | D1, D2              |
| **Multiple arrays**  | No           | Yes          | No                  |
| **Human-readable**   | No           | No           | Yes                 |
| **Compressed input** | No           | Built-in     | `.gz`, `.zip`       |

<seealso style="cards">
<category ref="api-docs">
  <a href="ndarray-class.md" summary="NDArray class — properties, operators, and methods."/>
  <a href="scalars.md" summary="Supported numeric and complex element types."/>
</category>
<category ref="user-guide">
  <a href="input-and-output.md" summary="User guide for I/O with practical examples."/>
</category>
</seealso>