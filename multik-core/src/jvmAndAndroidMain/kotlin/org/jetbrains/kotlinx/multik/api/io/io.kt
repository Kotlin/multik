package org.jetbrains.kotlinx.multik.api.io

import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.io.File
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.extension
import kotlin.io.path.notExists

/** Supported file formats for ndarray I/O, identified by file extension. */
@PublishedApi
internal enum class FileFormats(val extension: String) {
    NPY("npy"),
    NPZ("npz"),
    CSV("csv"),
}

/**
 * Reads an [NDArray] from the file at [fileName].
 *
 * The format is inferred from the file extension (`.npy` or `.csv`). For NPZ use [readNPZ].
 *
 * @param T the element type.
 * @param D the dimension type.
 * @param fileName path to the file including extension.
 * @throws NoSuchFileException if the file does not exist.
 * @see readNPY
 * @see readCSV
 */
public inline fun <reified T : Any, reified D : Dimension> Multik.read(fileName: String): NDArray<T, D> =
    this.read(Path(fileName))

/**
 * Reads an [NDArray] from the given [file].
 *
 * @see [read]
 */
public inline fun <reified T : Any, reified D : Dimension> Multik.read(file: File): NDArray<T, D> = this.read(file.path)

/**
 * Reads an [NDArray] from the given [path].
 *
 * @see [read]
 */
public inline fun <reified T : Any, reified D : Dimension> Multik.read(path: Path): NDArray<T, D> =
    this.read(path, DataType.ofKClass(T::class), dimensionClassOf<D>())

/**
 * Reads an [NDArray] from the given [path] with explicit [dtype] and dimension [dim].
 *
 * The format is inferred from the file extension:
 * - `.npy` — reads NumPy binary format (numeric types only, no complex).
 * - `.csv` — reads comma-separated values (D1 and D2 only).
 *
 * For NPZ archives, use [readNPZ] instead.
 *
 * @param path path to the file.
 * @param dtype the expected element data type.
 * @param dim the expected dimension.
 * @throws NoSuchFileException if the file does not exist.
 * @throws IllegalArgumentException if the format is unsupported or incompatible with [dtype]/[dim].
 */
public fun <T : Any, D : Dimension> Multik.read(path: Path, dtype: DataType, dim: D): NDArray<T, D> {
    if (path.notExists()) throw NoSuchFileException(path.toFile())
    return when (path.extension) {
        FileFormats.NPY.extension -> {
            require(dtype.isNumber()) { "NPY format does not support ${dtype.name}: only real numeric types are supported." }
            this.readNPY(path, dtype, dim)
        }

        FileFormats.CSV.extension -> {
            require(dim.d <= 2) { "CSV format supports 1- and 2-dimensional arrays, but got dimension ${dim.d}." }
            this.readRaw(path.toFile(), dtype, dim as Dim2) as NDArray<T, D>
        }

        else -> throw IllegalArgumentException(
            "Cannot read an ndarray from a `.${path.extension}` file. Supported formats: `npy`, `csv`. " +
                "For `npz` archives use `mk.readNPZ`."
        )
    }
}

/**
 * Writes the given [ndarray] to a file at [fileName].
 *
 * The format is inferred from the file extension (`.npy` or `.csv`).
 *
 * @param fileName path to the output file including extension.
 * @param ndarray the array to write.
 * @see writeNPY
 * @see writeCSV
 */
public fun Multik.write(fileName: String, ndarray: NDArray<*, *>): Unit =
    this.write(Path(fileName), ndarray)

/**
 * Writes the given [ndarray] to the specified [file].
 *
 * @see [write]
 */
public fun Multik.write(file: File, ndarray: NDArray<*, *>): Unit =
    this.write(file.toPath(), ndarray)

/**
 * Writes the given [ndarray] to the specified [path].
 *
 * The format is inferred from the file extension:
 * - `.npy` — NumPy binary format (numeric types only, no complex).
 * - `.csv` — comma-separated values (D1 and D2 only).
 *
 * @param path path to the output file.
 * @param ndarray the array to write.
 * @throws IllegalArgumentException if the format is unsupported or incompatible with the array's type or dimension.
 */
public fun Multik.write(path: Path, ndarray: NDArray<*, *>): Unit =
    when (path.extension) {
        FileFormats.NPY.extension -> {
            require(ndarray.dtype.isNumber()) {
                "NPY format does not support ${ndarray.dtype.name}: only real numeric types are supported."
            }
            this.writeNPY(path, ndarray as NDArray<out Number, *>)
        }

        FileFormats.CSV.extension -> {
            require(ndarray.dim.d <= 2) {
                "CSV format supports 1- and 2-dimensional arrays, but got dimension ${ndarray.dim.d}."
            }
            this.writeCSV(path.toFile(), ndarray as NDArray<*, out Dim2>)
        }

        else -> throw IllegalArgumentException(
            "Cannot write an ndarray to a `.${path.extension}` file. Supported formats: `npy`, `csv`."
        )
    }
