package org.jetbrains.kotlinx.multik.api

import org.jetbrains.kotlinx.multik.ndarray.data.Dim2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import java.nio.ByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Paths

@JvmName("readWithTypeAndDimension")
public inline fun <reified T : Number, reified D : Dim2> Multik.read(
    file: String,
    delimiter: Char = ','
): NDArray<T, D> {
//    val reader = BufferedReader(InputStreamReader(FileInputStream(file), "UTF-8"))

    val path = Paths.get(file)
    val buffer = ByteBuffer.allocateDirect(Files.size(path).toInt())
    val fileChannel = FileChannel.open(path)

    println(fileChannel.read(buffer))
//    val data = buffer.asIntBuffer()

    println(buffer.getInt())


//    val format = CSVFormat.DEFAULT.withDelimiter(delimiter)
//    val csvParser = format.parse(reader)
//    val records = csvParser.records
//
//    println(records.size)
//    for (r in records) {
//        println(r)
//        println(r.size())
//
//    }
    return mk.empty(2)
}

//@JvmName("readWithoutTypeAndDimension")
//public fun Multik.read(file: String, delimiter: String): NDArray<Double, DN> {
//
//}