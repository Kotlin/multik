package samples.docs.apiDocs

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.operations.all
import org.jetbrains.kotlinx.multik.ndarray.operations.and
import org.jetbrains.kotlinx.multik.ndarray.operations.any
import org.jetbrains.kotlinx.multik.ndarray.operations.append
import org.jetbrains.kotlinx.multik.ndarray.operations.asSequence
import org.jetbrains.kotlinx.multik.ndarray.operations.associate
import org.jetbrains.kotlinx.multik.ndarray.operations.associateBy
import org.jetbrains.kotlinx.multik.ndarray.operations.associateWith
import org.jetbrains.kotlinx.multik.ndarray.operations.average
import org.jetbrains.kotlinx.multik.ndarray.operations.chunked
import org.jetbrains.kotlinx.multik.ndarray.operations.clip
import org.jetbrains.kotlinx.multik.ndarray.operations.contains
import org.jetbrains.kotlinx.multik.ndarray.operations.count
import org.jetbrains.kotlinx.multik.ndarray.operations.distinct
import org.jetbrains.kotlinx.multik.ndarray.operations.distinctBy
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.drop
import org.jetbrains.kotlinx.multik.ndarray.operations.dropWhile
import org.jetbrains.kotlinx.multik.ndarray.operations.filter
import org.jetbrains.kotlinx.multik.ndarray.operations.filterIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.filterNot
import org.jetbrains.kotlinx.multik.ndarray.operations.find
import org.jetbrains.kotlinx.multik.ndarray.operations.findLast
import org.jetbrains.kotlinx.multik.ndarray.operations.first
import org.jetbrains.kotlinx.multik.ndarray.operations.firstOrNull
import org.jetbrains.kotlinx.multik.ndarray.operations.flatMap
import org.jetbrains.kotlinx.multik.ndarray.operations.fold
import org.jetbrains.kotlinx.multik.ndarray.operations.foldIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.forEach
import org.jetbrains.kotlinx.multik.ndarray.operations.forEachMultiIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.groupNDArrayBy
import org.jetbrains.kotlinx.multik.ndarray.operations.groupingNDArrayBy
import org.jetbrains.kotlinx.multik.ndarray.operations.indexOf
import org.jetbrains.kotlinx.multik.ndarray.operations.indexOfFirst
import org.jetbrains.kotlinx.multik.ndarray.operations.indexOfLast
import org.jetbrains.kotlinx.multik.ndarray.operations.intersect
import org.jetbrains.kotlinx.multik.ndarray.operations.joinToString
import org.jetbrains.kotlinx.multik.ndarray.operations.last
import org.jetbrains.kotlinx.multik.ndarray.operations.lastIndexOf
import org.jetbrains.kotlinx.multik.ndarray.operations.lastOrNull
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.mapIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.mapMultiIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import org.jetbrains.kotlinx.multik.ndarray.operations.maxBy
import org.jetbrains.kotlinx.multik.ndarray.operations.maxWith
import org.jetbrains.kotlinx.multik.ndarray.operations.maximum
import org.jetbrains.kotlinx.multik.ndarray.operations.min
import org.jetbrains.kotlinx.multik.ndarray.operations.minBy
import org.jetbrains.kotlinx.multik.ndarray.operations.minWith
import org.jetbrains.kotlinx.multik.ndarray.operations.minimum
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.onEach
import org.jetbrains.kotlinx.multik.ndarray.operations.or
import org.jetbrains.kotlinx.multik.ndarray.operations.partition
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.plusAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.reduce
import org.jetbrains.kotlinx.multik.ndarray.operations.reduceIndexed
import org.jetbrains.kotlinx.multik.ndarray.operations.repeat
import org.jetbrains.kotlinx.multik.ndarray.operations.reversed
import org.jetbrains.kotlinx.multik.ndarray.operations.scan
import org.jetbrains.kotlinx.multik.ndarray.operations.sorted
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.timesAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.toArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toCollection
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toHashSet
import org.jetbrains.kotlinx.multik.ndarray.operations.toIntArray
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import org.jetbrains.kotlinx.multik.ndarray.operations.toListD2
import org.jetbrains.kotlinx.multik.ndarray.operations.toMutableList
import org.jetbrains.kotlinx.multik.ndarray.operations.toSet
import org.jetbrains.kotlinx.multik.ndarray.operations.toType
import org.jetbrains.kotlinx.multik.ndarray.operations.unaryMinus
import org.jetbrains.kotlinx.multik.ndarray.operations.windowed
import kotlin.test.Ignore
import kotlin.test.Test

class ArrayOperations {

    // === Arithmetic Operations ===

    @Test
    fun addition_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        val b = mk.ndarray(mk[10, 20, 30])

        val c = a + b        // [11, 22, 33]
        val d = a + 100      // [101, 102, 103]
        val e = 5 + a        // [6, 7, 8]
        // SampleEnd
    }

    @Test
    fun subtraction_example() {
        // SampleStart
        val a = mk.ndarray(mk[10, 20, 30])
        val b = mk.ndarray(mk[1, 2, 3])

        val c = a - b        // [9, 18, 27]
        val d = a - 5        // [5, 15, 25]
        val e = 100 - a      // [90, 80, 70]
        // SampleEnd
    }

    @Test
    fun multiplication_example() {
        // SampleStart
        val a = mk.ndarray(mk[2, 3, 4])
        val b = mk.ndarray(mk[10, 20, 30])

        val c = a * b        // [20, 60, 120]
        val d = a * 3        // [6, 9, 12]
        // SampleEnd
    }

    @Test
    fun division_example() {
        // SampleStart
        val a = mk.ndarray(mk[10.0, 20.0, 30.0])

        val b = a / 2.0      // [5.0, 10.0, 15.0]
        // SampleEnd
    }

    @Test
    fun unary_negation_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, -2, 3])
        val b = -a           // [-1, 2, -3]
        // SampleEnd
    }

    @Test
    fun in_place_arithmetic_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        a += 10              // a is now [11, 12, 13]
        a *= 2               // a is now [22, 24, 26]
        // SampleEnd
    }

    // === Logical Operations ===

    @Test
    fun and_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 0, 3, 0])
        val b = mk.ndarray(mk[1, 1, 0, 0])

        val c = a and b      // [1, 0, 0, 0]
        // SampleEnd
    }

    @Test
    fun or_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 0, 3, 0])
        val b = mk.ndarray(mk[1, 1, 0, 0])

        val c = a or b       // [1, 1, 1, 0]
        // SampleEnd
    }

    // === Comparison Operations ===

    @Test
    fun minimum_comparison_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 5, 3])
        val b = mk.ndarray(mk[4, 2, 6])

        val c = a.minimum(b)  // [1, 2, 3]
        // SampleEnd
    }

    @Test
    fun maximum_comparison_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 5, 3])
        val b = mk.ndarray(mk[4, 2, 6])

        val c = a.maximum(b)  // [4, 5, 6]
        // SampleEnd
    }

    // === Predicates ===

    @Test
    fun all_example() {
        // SampleStart
        val a = mk.ndarray(mk[2, 4, 6, 8])

        a.all { it % 2 == 0 }   // true
        a.all { it > 5 }         // false

        val empty = mk.ndarray(mk[0]).drop(1)
        empty.all { it > 0 }     // true (vacuously true)
        // SampleEnd
    }

    @Test
    fun any_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4])

        a.any()              // true
        a.any { it > 3 }     // true
        a.any { it > 10 }    // false
        // SampleEnd
    }

    @Test
    fun contains_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])

        a.contains(3)    // true
        3 in a           // true  (operator syntax)
        10 in a          // false
        // SampleEnd
    }

    // === Traversal ===

    @Test
    fun forEach_example() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])

        a.forEach { print("$it ") }
        // 1 2 3 4

        a.forEachMultiIndexed { idx, v ->
            println("${idx.toList()} -> $v")
        }
        // [0, 0] -> 1
        // [0, 1] -> 2
        // [1, 0] -> 3
        // [1, 1] -> 4
        // SampleEnd
    }

    @Test
    fun onEach_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])

        val result = a
            .onEach { print("$it ") }  // prints: 1 2 3
            .map { it * 2 }
        // result: [2, 4, 6]
        // SampleEnd
    }

    // === Transformation ===

    @Test
    @Ignore
    fun map_example() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])

        val doubled = a.map { it * 2 }
        // [[2, 4],
        //  [6, 8]]

        val b = mk.ndarray(mk[10, 20, 30])
        b.mapIndexed { i, v -> v + i }
        // [10, 21, 32]

        a.mapMultiIndexed { idx, v -> idx.sum() + v }
        // [[1, 3],
        //  [4, 6]]
        // SampleEnd
    }

    @Test
    fun flatMap_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])

        val result = a.flatMap { listOf(it, it * 10) }
        // [1, 10, 2, 20, 3, 30]
        // SampleEnd
    }

    @Test
    fun scan_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4])

        a.scan(0) { acc, v -> acc + v }
        // [0, 1, 3, 6, 10]  — cumulative sum with initial 0
        // SampleEnd
    }

    // === Filtering ===

    @Test
    fun filter_example() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

        a.filter { it > 3 }                       // [4, 5, 6]
        a.filterNot { it > 3 }                    // [1, 2, 3]

        val b = mk.ndarray(mk[10, 20, 30, 40])
        b.filterIndexed { i, v -> i % 2 == 0 }    // [10, 30]
        // SampleEnd
    }

    @Test
    fun find_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])

        a.find { it > 3 }       // 4
        a.findLast { it > 3 }   // 5
        a.find { it > 10 }      // null
        // SampleEnd
    }

    @Test
    fun distinct_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 2, 3, 1, 4])

        a.distinct()                  // [1, 2, 3, 4]
        a.distinctBy { it % 2 }      // [1, 2]  (first odd, first even)
        // SampleEnd
    }

    // === Aggregation ===

    @Test
    fun fold_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4])

        a.fold(0) { acc, v -> acc + v }                // 10
        a.fold("") { acc, v -> "$acc$v" }              // "1234"

        val b = mk.ndarray(mk[10, 20, 30])
        b.foldIndexed(0) { i, acc, v -> acc + i * v }  // 0*10 + 1*20 + 2*30 = 80
        // SampleEnd
    }

    @Test
    fun reduce_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4])

        a.reduce { acc, v -> acc + v }        // 10
        a.reduce { acc, v -> acc * v }        // 24

        a.reduceIndexed { i, acc, v -> acc + i * v }
        // 1 + 1*2 + 2*3 + 3*4 = 21
        // SampleEnd
    }

    // TODO
//    @Test
//    fun sum_example() {
//        // SampleStart
//        val a = mk.ndarray(mk[1, 2, 3, 4, 5])
//
//        a.sum()               // 15
//        a.sumBy { it * it }   // 55 (sum of squares)
//
//        val b = mk.ndarray(mk[1.5, 2.5, 3.0])
//        b.sum()                      // 7.0
//        // SampleEnd
//    }

    @Test
    fun average_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])
        a.average()  // 3.0

        val b = mk.ndarray(mk[1.5, 2.5])
        b.average()  // 2.0
        // SampleEnd
    }

    @Test
    fun count_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])

        a.count()            // 5
        a.count { it > 3 }   // 2
        // SampleEnd
    }

    @Test
    fun max_example() {
        // SampleStart
        val a = mk.ndarray(mk[3, 1, 4, 1, 5])

        a.max()                        // 5
        a.maxBy { -it }                // 1  (largest by negated value)
        a.maxWith(compareBy { it })    // 5
        // SampleEnd
    }

    @Test
    fun min_example() {
        // SampleStart
        val a = mk.ndarray(mk[3, 1, 4, 1, 5])

        a.min()                        // 1
        a.minBy { -it }                // 5  (smallest by negated value)
        a.minWith(compareBy { it })    // 1
        // SampleEnd
    }

    @Test
    fun maximum_example() {
        // SampleStart
        val a = mk.ndarray(mk[3, 1, 4])
        val b = mk.ndarray(mk[1, 5, 2])

        a.maximum(b)  // [3, 5, 4]
        // SampleEnd
    }

    @Test
    fun minimum_example() {
        // SampleStart
        val a = mk.ndarray(mk[3, 1, 4])
        val b = mk.ndarray(mk[1, 5, 2])

        a.minimum(b)  // [1, 1, 2]
        // SampleEnd
    }

    // === Search ===

    @Test
    fun first_example() {
        // SampleStart
        val a = mk.ndarray(mk[10, 20, 30])

        a.first()                // 10
        a.first { it > 15 }      // 20
        a.firstOrNull { it > 50 } // null
        // SampleEnd
    }

    @Test
    fun last_example() {
        // SampleStart
        val a = mk.ndarray(mk[10, 20, 30])

        a.last()                 // 30
        a.last { it < 25 }       // 20
        a.lastOrNull { it > 50 } // null
        // SampleEnd
    }

    @Test
    fun indexOf_example() {
        // SampleStart
        val a = mk.ndarray(mk[10, 20, 30, 20, 10])

        a.indexOf(20)                // 1
        a.lastIndexOf(20)            // 3
        a.indexOfFirst { it > 15 }   // 1
        a.indexOfLast { it > 15 }    // 3
        a.indexOf(99)                // -1
        // SampleEnd
    }

    // === Ordering ===

    @Test
    fun sorted_example() {
        // SampleStart
        val a = mk.ndarray(mk[3, 1, 4, 1, 5, 9])
        a.sorted()  // [1, 1, 3, 4, 5, 9]

        val m = mk.ndarray(mk[mk[3, 1], mk[4, 2]])
        m.sorted()
        // [[1, 2],
        //  [3, 4]]
        // SampleEnd
    }

    @Test
    fun reversed_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])
        a.reversed()  // [5, 4, 3, 2, 1]

        val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        m.reversed()
        // [[4, 3],
        //  [2, 1]]
        // SampleEnd
    }

    // === Partitioning ===

    @Test
    fun partition_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6])

        val (evens, odds) = a.partition { it % 2 == 0 }
        // evens: [2, 4, 6]
        // odds:  [1, 3, 5]
        // SampleEnd
    }

    @Test
    fun chunked_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])

        val c = a.chunked(2)
        // [[1, 2],
        //  [3, 4],
        //  [5, 0]]   — last chunk padded with 0
        // SampleEnd
    }

    @Test
    @Ignore
    fun windowed_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])

        a.windowed(3)
        // [[1, 2, 3],
        //  [2, 3, 4],
        //  [3, 4, 5]]

        a.windowed(3, step = 2)
        // [[1, 2, 3],
        //  [3, 4, 5]]
        // SampleEnd
    }

    @Test
    fun drop_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])

        a.drop(2)                    // [3, 4, 5]
        a.dropWhile { it < 3 }      // [3, 4, 5]
        // SampleEnd
    }

    // === Grouping ===

    @Test
    fun groupNDArrayBy_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6])

        val groups = a.groupNDArrayBy { if (it % 2 == 0) "even" else "odd" }
        // "odd"  -> [1, 3, 5]
        // "even" -> [2, 4, 6]

        val grouping = a.groupingNDArrayBy { it % 3 }
        grouping.eachCount()  // {1=2, 2=2, 0=2}
        // SampleEnd
    }

    @Test
    fun associate_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])

        a.associate { it to it * it }       // {1=1, 2=4, 3=9}
        a.associateBy { it * 10 }           // {10=1, 20=2, 30=3}
        a.associateWith { it.toString() }   // {1="1", 2="2", 3="3"}
        // SampleEnd
    }

    @Test
    fun intersect_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])

        a intersect listOf(3, 4, 5, 6, 7)    // {3, 4, 5}
        a intersect listOf(10, 20)           // {}
        // SampleEnd
    }

    // === Conversion ===

    @Test
    fun toList_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        a.toList()       // [1, 2, 3]
        a.toMutableList() // mutable [1, 2, 3]

        val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        m.toList()       // [1, 2, 3, 4]  (flat)
        m.toListD2()     // [[1, 2], [3, 4]]  (nested)
        // SampleEnd
    }

    @Test
    fun toCollection_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 2, 3, 3, 3])

        a.toSet()                            // {1, 2, 3}
        a.toHashSet()                        // HashSet {1, 2, 3}

        val list = mutableListOf(0)
        a.toCollection(list)                 // [0, 1, 2, 2, 3, 3, 3]
        // SampleEnd
    }

    @Test
    fun toPrimitiveArray_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        a.toIntArray()       // IntArray [1, 2, 3]

        val b = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        b.toDoubleArray()    // DoubleArray [1.0, 2.0, 3.0, 4.0]  (flat)
        // SampleEnd
    }

    @Test
    fun toArray_example() {
        // SampleStart
        val m = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

        val arr: Array<IntArray> = m.toArray()
        // arr[0] = IntArray [1, 2, 3]
        // arr[1] = IntArray [4, 5, 6]
        // SampleEnd
    }

    @Test
    fun asSequence_example() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])

        val result = a.asSequence()
            .filter { it > 2 }
            .map { it * 10 }
            .toList()
        // [30, 40, 50, 60]
        // SampleEnd
    }

    @Test
    fun toType_example() {
        // SampleStart
        val ints = mk.ndarray(mk[1, 2, 3])

        val doubles = ints.toType<Int, Double, D1>()
        // D1Array<Double> [1.0, 2.0, 3.0]

        val floats = ints.toType<Int, Float, D1>()
        // D1Array<Float> [1.0, 2.0, 3.0]
        // SampleEnd
    }

    // === Transformation Operations ===

    @Test
    fun append_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])

        // Append individual elements
        a.append(4, 5)               // [1, 2, 3, 4, 5]

        // Append another array (flattened)
        val b = mk.ndarray(mk[10, 20])
        a append b                   // [1, 2, 3, 10, 20]

        // Append along an axis
        val m1 = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val m2 = mk.ndarray(mk[mk[5, 6]])
        m1.append(m2, axis = 0)
        // [[1, 2],
        //  [3, 4],
        //  [5, 6]]
        // SampleEnd
    }

    @Test
    fun repeat_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        a.repeat(3) // [1, 2, 3, 1, 2, 3, 1, 2, 3]

        val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        m.repeat(2) // [1, 2, 3, 4, 1, 2, 3, 4]
        // SampleEnd
    }

    @Test
    fun clip_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 5, 10, 15, 20])
        a.clip(5, 15) // [5, 5, 10, 15, 15]

        val m = mk.ndarray(mk[mk[0.0, 0.5], mk[1.0, 1.5]])
        m.clip(0.2, 1.0)
        // [[0.2, 0.5],
        //  [1.0, 1.0]]
        // SampleEnd
    }

    @Test
    fun joinToString_example() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3, 4, 5])

        a.joinToString()                           // "1, 2, 3, 4, 5"
        a.joinToString(separator = " | ")          // "1 | 2 | 3 | 4 | 5"
        a.joinToString(prefix = "[", postfix = "]") // "[1, 2, 3, 4, 5]"
        a.joinToString(limit = 3)                  // "1, 2, 3, ..."
        a.joinToString { (it * 10).toString() }    // "10, 20, 30, 40, 50"
        // SampleEnd
    }
}
