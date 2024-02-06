package org.jetbrains.kotlinx.multik.api.io

import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.io.File
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.extension
import kotlin.io.path.notExists

@PublishedApi
internal enum class FileFormats(val extension: String) {
    NPY("npy"),
    NPZ("npz"),
    CSV("csv"),
}

public inline fun <reified T : Any, reified D : Dimension> Multik.read(fileName: String): NDArray<T, D> =
    this.read(Path(fileName))

public inline fun <reified T : Any, reified D : Dimension> Multik.read(file: File): NDArray<T, D> = this.read(file.path)

public inline fun <reified T : Any, reified D : Dimension> Multik.read(path: Path): NDArray<T, D> =
    this.read(path, DataType.ofKClass(T::class), dimensionClassOf<D>())

public fun <T : Any, D : Dimension> Multik.read(path: Path, dtype: DataType, dim: D): NDArray<T, D> {
    if (path.notExists()) throw NoSuchFileException(path.toFile())
    return when (path.extension) {
        FileFormats.NPY.extension -> {
            if (dtype.isComplex()) throw Exception("NPY format only supports Number types")
            this.readNPY(path, dtype, dim)
        }

        FileFormats.CSV.extension -> {
            if (dim.d > 2) throw Exception("CSV format only supports 1 and 2 dimensions")
            this.readRaw(path.toFile(), dtype, dim as Dim2) as NDArray<T, D>
        }

        else -> throw Exception("Format ${path.extension} does not support reading ndarrays. If it is `npz` format, try `mk.readNPZ`")
    }
}

public fun Multik.write(fileName: String, ndarray: NDArray<*, *>): Unit =
    this.write(Path(fileName), ndarray)

public fun Multik.write(file: File, ndarray: NDArray<*, *>): Unit =
    this.write(file.toPath(), ndarray)

public fun Multik.write(path: Path, ndarray: NDArray<*, *>): Unit =
    when (path.extension) {
        FileFormats.NPY.extension -> {
            require(ndarray.dtype != DataType.ComplexFloatDataType || ndarray.dtype != DataType.ComplexFloatDataType) {
                "NPY format does not support complex numbers."
            }
            this.writeNPY(path, ndarray as NDArray<out Number, *>)
        }

        FileFormats.CSV.extension -> {
            require(ndarray.dim.d < 2) { "Expected array of dimension less than 2, but got array of dimension ${ndarray.dim.d}." }
            this.writeCSV(path.toFile(), ndarray as NDArray<*, out Dim2>)
        }

        else -> throw Exception("Unknown format `${path.extension}`. Please use one of the supported formats: `npy`, `csv`.")
    }
