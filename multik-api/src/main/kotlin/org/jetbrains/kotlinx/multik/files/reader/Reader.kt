package org.jetbrains.kotlinx.multik.files.reader

import java.io.File
import java.io.IOException
import java.io.Reader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths

internal enum class Formats {
    CSV,
    NPY
}

public enum class QuoteMode {
    None
}

internal fun getFormat(file: File): Formats? = when (file.extension.lowercase()) {
    "csv" -> Formats.CSV
    "npy" -> Formats.NPY
    else -> null
}

private const val CR: Char = '\r'
private const val LF: Char = '\n'

public class ReaderCSV(
    public val delimiter: Char = ',',
    public val quote: Char = '"',
    public val quoteMode: QuoteMode = QuoteMode.None,
    public val comment: Char = '#',
    public val skipEmptyLines: Boolean = true
) {
    private lateinit var channel: FileChannel
    private lateinit var path: Path
    private var fileSize: Long = 0L

    init {
        require(delimiter != CR || delimiter != LF) { "delimiter must not be a newline character." }
        require(quote != CR || quote != LF) { "quote must not be a newline character." }
        require(comment != CR || comment != LF) { "comment must not be a newline character." }
        require(delimiter != quote || delimiter != comment || quote != comment) {
            "delimiter, quote and comment characters must not match: delimiter=$delimiter, quote=$quote, comment=$comment"
        }
    }

    public fun read(file: String, cache: Boolean = false): Records {
        require(file.isNotEmpty()) { "Path must not be empty." }
        path = Paths.get(file)
        require(Files.exists(path)) { "Path must exist." }
        channel = FileChannel.open(path)
        fileSize = Files.size(path)

        val records = Records(channel, fileSize)
        // body
        if (cache) {
            records.keepRows()
            close()
        }

        return records
    }

    internal fun close() {
        if (channel.isOpen) {
            channel.close()
        }
    }
}

private class Buffer internal constructor(private val reader: Reader) {
    private val READ_SIZE = 8192
    private val BUFFER_SIZE = READ_SIZE
    private val MAX_BUFFER_SIZE = 8 * 1024 * 1024
    var buf = CharArray(BUFFER_SIZE)
    var len = 0
    var begin = 0
    var pos = 0

    /**
     * Reads data from the underlying reader and manages the local buffer.
     *
     * @return `true`, if EOD reached.
     * @throws IOException if a read error occurs
     */
    internal fun fetchData(): Boolean {
        if (begin < pos) {
            // we have data that can be relocated
            if (READ_SIZE > buf.size - pos) {
                // need to relocate data in buffer -- not enough capacity left
                val lenToCopy = pos - begin
                if (READ_SIZE > buf.size - lenToCopy) {
                    // need to relocate data in new, larger buffer
                    buf = extendAndRelocate(buf, begin)
                } else {
                    // relocate data in existing buffer
                    System.arraycopy(buf, begin, buf, 0, lenToCopy)
                }
                pos -= begin
                begin = 0
            }
        } else {
            // all data was consumed -- nothing to relocate
            begin = 0
            pos = begin
        }
        val cnt = reader.read(buf, pos, READ_SIZE)
        if (cnt == -1) {
            return true
        }
        len = pos + cnt
        return false
    }

    private fun extendAndRelocate(buf: CharArray, begin: Int): CharArray {
        val newBufferSize = buf.size * 2
        if (newBufferSize > MAX_BUFFER_SIZE) {
            throw IOException(
                "Maximum buffer size "
                    + MAX_BUFFER_SIZE + " is not enough to read data"
            )
        }
        val newBuf = CharArray(newBufferSize)
        System.arraycopy(buf, begin, newBuf, 0, buf.size - begin)
        return newBuf
    }
}

public class Records internal constructor(private val channel: FileChannel, fileSize: Long) :
    Iterable<StringRow> {
    public val recordList: MutableList<StringRow> = ArrayList(32)
    private val buffer: DataBuffer = DataBuffer(fileSize)


    private var finished: Boolean = false
    private val listRow = ArrayList<String>()


    public fun add(row: StringRow) {

    }

    internal fun keepRows() {
        while (true) {
            val row = getNextRow() ?: break
            recordList.add(row)
        }
    }

    private fun getNextRow(): StringRow? {
        while (true) {
            val row = receiveNextRow() ?: return null
            if (row.isEmpty()) {
                continue
            }
            return row
        }
    }

    private fun receiveNextRow(): StringRow? {
        val listStringRow = mutableListOf<String>()
        if (finished) {
            return null
        }

        do {
            if (buffer.len == buffer.pos) {
                if (buffer.getData()) {
                    if (buffer.begin < buffer.pos) {

                        listStringRow.add(
                            String(buffer.data.array(), buffer.begin, buffer.pos - buffer.begin, Charsets.UTF_8)
//                        String(buffer.data, buffer.begin, buffer.pos - buffer.begin)
                        )
                    }
                    finished = true
                    break
                }
            }
        } while (consume(listStringRow, buffer.data))
        return StringRow(listStringRow)
    }

    private fun consume(listStringRow: MutableList<String>, data: ByteBuffer): Boolean {
        var lPos = buffer.pos
        var lBegin = buffer.begin
        val length = buffer.data.limit()
        var moreDataNeeded = true
        val charset = Charsets.UTF_8
        while (lPos < length) {
            val c = data[lPos++].toChar()
            if (c == ',') {
                listStringRow.add(
                    String(buffer.data.array(), buffer.begin, lPos - lBegin - 1, charset)
                )
                lBegin = lPos
            } else if (c == CR) {
                listStringRow.add(
                    String(buffer.data.array(), buffer.begin, lPos - lBegin - 1, charset)
                )
                lBegin = lPos
                moreDataNeeded = false
                break
            } else if (c == LF) {
                listStringRow.add(
                    String(buffer.data.array(), buffer.begin, lPos - lBegin - 1, charset)
                )
                lBegin = lPos
                moreDataNeeded = false
                break
            } else if (c == '#') {
                lBegin = lPos
                continue
            } else if (c == '"') {
                continue
            } else {
                while (lPos < length) {
                    val lookAhead = data[lPos++].toChar()
                    if (lookAhead == ',' || lookAhead == LF || lookAhead == CR) {
                        lPos--
                        break
                    }
                }
            }
        }

        buffer.pos = lPos
        buffer.begin = lBegin
        return moreDataNeeded
    }


    override fun iterator(): Iterator<StringRow> =
        if (recordList.isNotEmpty()) {
            recordList.iterator()
        } else {
            RecordsIterator()
        }

    private inner class RecordsIterator : Iterator<StringRow> {
        private var currentRow: StringRow? = null
        private var received: Boolean = false

        override fun hasNext(): Boolean {
//            if (!this@Records.channel.isOpen) {
//            if (!this@Records.channel.ready()) {
//                return false
//            }
            if (!received) {
                this.currentRow = receive()
            }

            return if (this.currentRow != null) {
                true
            } else {
                this@Records.channel.close()
                false
            }
        }

        override fun next(): StringRow {
            if (!this@Records.channel.isOpen) {
                throw NoSuchElementException("Reader has been closed.")
            }
            if (!received) {
                this.currentRow = receive()
            }
            if (this.currentRow == null) {
                throw NoSuchElementException("No more fields available.")
            }
            received = false

            return this.currentRow!!
        }

        private fun receive(): StringRow? {
            try {
                received = true
                return this@Records.getNextRow()
            } catch (e: IOException) {
                throw IllegalStateException(e.localizedMessage, e)
            }
        }
    }

    private inner class DataBuffer(fileSize: Long) {
        // todo size for bytebuffer!
        private val READ_SIZE = 8129
        private val initialCapacity = if (fileSize < Int.MAX_VALUE) fileSize.toInt() else Int.MAX_VALUE
        private var readPos: Long = 0

        var data: ByteBuffer = ByteBuffer.allocate(initialCapacity).order(ByteOrder.nativeOrder())

        //        var data: CharBuffer = ByteBuffer.allocate(INITIAL_CAPACITY).order(ByteOrder.nativeOrder()).asCharBuffer()
        var len: Int = 0
        var begin: Int = 0
        var pos: Int = 0

        fun getData(): Boolean {
            data.clear()
            val count = channel.read(data, readPos)
            return if (count == -1) {
                true
            } else {
                readPos += count.toLong()
                len = data.limit()
                false
            }
//            val capacity = data.capacity()
//
//            if ((begin < pos) && (READ_SIZE > capacity - pos)) {
//                data = increase(data)
//                pos -= begin
//                begin = 0
//                data.clear()
//            } else {
//                begin = 0
//                pos = begin
//            }
//            val count = channel.read(data, pos.toLong())
//            return if (count < -1) {
//                true
//            } else {
//                len = pos + count
//                false
//            }
        }

        fun increase(data: ByteBuffer): ByteBuffer {
            val newBufferCapacity = data.capacity()
            if (newBufferCapacity > Int.MAX_VALUE) {
                throw IOException("Maximum buffer capacity ${Int.MAX_VALUE} is not enough to read data.")
            }
            val newData = ByteBuffer.allocate(newBufferCapacity).order(ByteOrder.nativeOrder())//.asCharBuffer()
            newData.put(data)
            newData.rewind()
            return newData
        }
    }
}

public interface Row {
    public val values: List<Any?>
    public val isComment: Boolean

    public fun isEmpty(): Boolean = values.isEmpty()
}

public class StringRow(override val values: List<String?>, override val isComment: Boolean = false) : Row
public class IntRow(override val values: List<Int?>, override val isComment: Boolean = false) : Row
public class LongRow(override val values: List<Long?>, override val isComment: Boolean = false) : Row
public class FloatRow(override val values: List<Float?>, override val isComment: Boolean = false) : Row
public class DoubleRow(override val values: List<Double?>, override val isComment: Boolean = false) : Row
public class CharRow(override val values: List<Char?>, override val isComment: Boolean = false) : Row