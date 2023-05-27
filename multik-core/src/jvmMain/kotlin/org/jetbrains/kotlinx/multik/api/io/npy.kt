package org.jetbrains.kotlinx.multik.api.io

import org.jetbrains.bio.npy.NpyArray
import org.jetbrains.bio.npy.NpyFile
import org.jetbrains.bio.npy.NpzFile
import org.jetbrains.kotlinx.multik.api.Multik
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.io.File
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.notExists

public inline fun <reified T : Number, reified D : Dimension> Multik.readNPY(fileName: String): NDArray<T, D> =
    readNPY(Path(fileName), DataType.ofKClass(T::class), dimensionClassOf<D>())

public inline fun <reified T : Number, reified D : Dimension> Multik.readNPY(file: File): NDArray<T, D> =
    readNPY(file.toPath(), DataType.ofKClass(T::class), dimensionClassOf<D>())

public inline fun <reified T : Number, reified D : Dimension> Multik.readNPY(path: Path): NDArray<T, D> {
    if (path.notExists()) throw NoSuchFileException(path.toFile())
    return readNPY(path, DataType.ofKClass(T::class), dimensionClassOf<D>())
}

public fun <T : Any, D : Dimension> Multik.readNPY(path: Path, dtype: DataType, dim: D): NDArray<T, D> {
    if (dtype.isComplex()) throw Exception("NPY format only supports Number types")
    val npyArray: NpyArray = NpyFile.read(path)
    require(npyArray.shape.size == dim.d) { "Not match dimensions: shape of npy array = ${npyArray.shape.joinToString()}, and dimension = ${dim.d}" }

    val data: MemoryView<T> = when (dtype) {
        DataType.DoubleDataType -> MemoryViewDoubleArray(npyArray.asDoubleArray())
        DataType.FloatDataType -> MemoryViewFloatArray(npyArray.asFloatArray())
        DataType.IntDataType -> MemoryViewIntArray(npyArray.asIntArray())
        DataType.LongDataType -> MemoryViewLongArray(npyArray.asLongArray())
        DataType.ShortDataType -> MemoryViewShortArray(npyArray.asShortArray())
        DataType.ByteDataType -> MemoryViewByteArray(npyArray.asByteArray())
        else -> throw Exception("not supported complex arrays")
    } as MemoryView<T>

    return NDArray(data, shape = npyArray.shape, dim = dim)
}

public fun Multik.readNPZ(fileName: String): List<NDArray<out Number, out DimN>> =
    readNPZ(Path(fileName))

public fun Multik.readNPZ(file: File): List<NDArray<*, out DimN>> =
    readNPZ(file.toPath())

public fun Multik.readNPZ(path: Path): List<NDArray<out Number, out DimN>> {
    return NpzFile.read(path).use {
        val entries = it.introspect()
        entries.map { entry ->
            val npyArray = it[entry.name]
            val data = when (entry.type.kotlin) {
                DataType.DoubleDataType.clazz -> MemoryViewDoubleArray(npyArray.asDoubleArray())
                DataType.FloatDataType.clazz -> MemoryViewFloatArray(npyArray.asFloatArray())
                DataType.IntDataType.clazz -> MemoryViewIntArray(npyArray.asIntArray())
                DataType.LongDataType.clazz -> MemoryViewLongArray(npyArray.asLongArray())
                DataType.ShortDataType.clazz -> MemoryViewShortArray(npyArray.asShortArray())
                DataType.ByteDataType.clazz -> MemoryViewByteArray(npyArray.asByteArray())
                else -> TODO()
            }
            NDArray(data, shape = npyArray.shape, dim = dimensionOf(npyArray.shape.size))
        }
    }
}

public fun <T : Number, D : Dimension> Multik.writeNPY(fileName: String, ndArray: NDArray<T, D>): Unit =
    this.writeNPY(Path(fileName), ndArray)

public fun <T : Number, D : Dimension> Multik.writeNPY(file: File, ndArray: NDArray<T, D>): Unit =
    this.writeNPY(file.toPath(), ndArray)

public fun <T : Number, D : Dimension> Multik.writeNPY(path: Path, ndArray: NDArray<T, D>): Unit {
    when (ndArray.dtype) {
        DataType.DoubleDataType -> NpyFile.write(path, ndArray.data.getDoubleArray(), ndArray.shape)
        DataType.FloatDataType -> NpyFile.write(path, ndArray.data.getFloatArray(), ndArray.shape)
        DataType.IntDataType -> NpyFile.write(path, ndArray.data.getIntArray(), ndArray.shape)
        DataType.LongDataType -> NpyFile.write(path, ndArray.data.getLongArray(), ndArray.shape)
        DataType.ShortDataType -> NpyFile.write(path, ndArray.data.getShortArray(), ndArray.shape)
        DataType.ByteDataType -> NpyFile.write(path, ndArray.data.getByteArray(), ndArray.shape)
        else -> TODO()
    }
}

public fun Multik.writeNPZ(path: Path, vararg ndArrays: NDArray<out Number, out Dimension>): Unit =
    this.writeNPZ(path, ndArrays.asList())

public fun Multik.writeNPZ(path: Path, ndArrays: List<NDArray<out Number, out Dimension>>): Unit {
    NpzFile.write(path).use {
        ndArrays.forEachIndexed { ind, array ->
            when (array.dtype) {
                DataType.DoubleDataType -> it.write("arr_$ind", array.data.getDoubleArray(), array.shape)
                DataType.FloatDataType -> it.write("arr_$ind", array.data.getFloatArray(), array.shape)
                DataType.IntDataType -> it.write("arr_$ind", array.data.getIntArray(), array.shape)
                DataType.LongDataType -> it.write("arr_$ind", array.data.getLongArray(), array.shape)
                DataType.ShortDataType -> it.write("arr_$ind", array.data.getShortArray(), array.shape)
                DataType.ByteDataType -> it.write("arr_$ind", array.data.getByteArray(), array.shape)
                else -> TODO()
            }
        }
    }
}
