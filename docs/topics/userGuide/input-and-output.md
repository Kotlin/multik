# Input and output

<!---IMPORT samples.docs.userGuide.InputAndOutput-->

<web-summary>
Learn how to move data in and out of Multik ndarrays.
Convert arrays to Kotlin collections or primitive arrays, and use JVM-only file I/O for CSV/NPY/NPZ.
</web-summary>

<card-summary>
Convert ndarrays to lists and primitive arrays, and use JVM I/O for CSV/NPY/NPZ.
</card-summary>

<link-summary>
Move data between Multik and Kotlin collections, and use JVM file I/O utilities.
</link-summary>

## Overview

Multik provides two groups of I/O tools:

- **In-memory conversions** for lists, sets, and primitive arrays (available on all platforms).
- **File I/O** for CSV/NPY/NPZ on the JVM.

## In-memory conversions

### Lists and collections

<!---FUN lists_and_collections-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
val list = a.toList()           // List<Int>
val mutable = a.toMutableList() // MutableList<Int>
val set = a.toSet()             // Set<Int>
```

<!---END-->

For nested lists (D2-D4):

<!---FUN nested_lists-->

```kotlin
val m = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
val list2 = m.toListD2() // List<List<Double>>
```

<!---END-->

### Primitive arrays

You can extract a flat primitive array from any ndarray:

<!---FUN primitive_arrays-->

```kotlin
val v = mk.ndarray(mk[1, 2, 3])
val ints = v.toIntArray() // IntArray
```

<!---END-->

For 2D-4D arrays, `toArray()` returns nested primitive arrays:

<!---FUN nested_primitive_arrays-->

```kotlin
val mat = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
val arr2 = mat.toArray() // Array<DoubleArray>
```

<!---END-->

Similar helpers exist for `Long`, `Float`, `Double`, and complex types.

## JVM file I/O

File-based I/O is available on the JVM only.
For NPY, you can use `mk.read` and `mk.write`.
For CSV and NPZ, use the format-specific helpers below.

| Format | Read         | Write         | Dimensions | Notes                                                  |
|--------|--------------|---------------|------------|--------------------------------------------------------|
| NPY    | `mk.read`    | `mk.write`    | Any        | Numeric types only.                                    |
| NPZ    | `mk.readNPZ` | `mk.writeNPZ` | Any        | Container for multiple NPY arrays; numeric types only. |
| CSV    | `mk.readCSV` | `mk.writeCSV` | 1D-2D      | Complex values supported with complex dtype.           |

### NPY

```kotlin
val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
mk.write("matrix.npy", a)

val loaded = mk.read<Double, D2>("matrix.npy")
```

### NPZ

```kotlin
mk.writeNPZ("bundle.npz", a)
val arrays = mk.readNPZ("bundle.npz")
```

### CSV

```kotlin
mk.writeCSV("matrix.csv", a)
val csv = mk.readCSV<Double, D2>("matrix.csv")
```

## Non-JVM platforms

On JS, WASM, and Native, use in-memory conversions and a platform-specific serialization library.
A common approach is to convert to primitive arrays or lists and serialize those.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="creating-multidimensional-arrays.md" summary="Create ndarrays from Kotlin collections and primitives."/>
  <a href="type-casting.md" summary="Convert ndarrays between numeric types."/>
  <a href="multik-on-JVM.md" summary="JVM-specific engine choices and setup notes."/>
</category>
<category ref="get-start">
  <a href="installation.md" summary="Add Multik dependencies to your project."/>
</category>
</seealso>