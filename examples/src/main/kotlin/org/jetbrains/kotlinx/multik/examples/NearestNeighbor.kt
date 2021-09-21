package org.jetbrains.kotlinx.multik.examples

import org.jetbrains.kotlinx.multik.api.abs
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.toList
import kotlin.random.Random

fun generateRandomPoints(size: Int = 10, min: Int = 0, max: Int = 1): NDArray<Double, D2> {
    val random = Random(1)
    return mk.d2array(size, 2) { random.nextDouble() }
}

class NearestNeighbor(private val distance: Int = 0) {
    private lateinit var xTrain: NDArray<Double, D2>
    private lateinit var yTrain: NDArray<Int, D1>

    fun train(x: NDArray<Double, D2>, y: NDArray<Int, D1>) {
        xTrain = x
        yTrain = y
    }

    fun predict(x: NDArray<Double, D2>): ArrayList<Int> {
        val predictions = ArrayList<Int>()
        for (x_i in 0 until x.shape[0]) {
            if (distance == 0) {
                val distances = mk.math.sumD2(abs(xTrain - x[x_i].broadcast(xTrain.shape[1])), axis = 1)
                val minIndex = mk.math.argMin(distances)

                predictions.add(yTrain[minIndex])
            } else if (distance == 1) {
                val distances = mk.math.sumD2(square(xTrain - x[x_i].broadcast(xTrain.shape[0])), axis = 1)
                val minIndex = mk.math.argMin(distances)

                predictions.add(yTrain[minIndex])
            }
        }

        return predictions
    }

    private fun <T : Number> MultiArray<T, D1>.broadcast(dim: Int): NDArray<T, D2> {
        val data = initMemoryView<T>(this.size * dim, this.dtype)
        for (i in 0 until dim) {
            var index = 0
            for (el in this) {
                data[index++] = el
            }
        }
        return D2Array(data, 0, intArrayOf(this.shape.first(), dim), dim = D2)
    }

    private fun <D : Dimension> square(x: MultiArray<Double, D>): NDArray<Double, D> {
        val data = initMemoryView<Double>(x.size, x.dtype)
        var index = 0
        for (element in x) {
            data[index++] = element * element
        }
        return NDArray(data, 0, x.shape.copyOf(), dim = x.dim)
    }
}

class Analysis(x: Array<NDArray<Double, D2>>, distance: Int) {
    val xTest: NDArray<Double, D2> = mk.d2array(40401, 2) { 0.0 }
//        private set
    lateinit var yTest: ArrayList<Int>
        private set

    private val nofClasses: Int = x.size
    private lateinit var range: Pair<Int, Int>

    private val classified: ArrayList<NDArray<Double, D1>> = ArrayList()
    private val nn: NearestNeighbor

    init {
//        val listY = ArrayList<Ndarray<Int, D1>>()
        val listY = ArrayList<Int>()
        for ((i, el) in x.indices.zip(x)) {
            listY.addAll(List(el.shape[0]) { 1 * i })
//            listY.add(mk.d1array(el.shape[0]) { 1 } * i)
        }
        val xt = concatenate(x, axis = 0)
        nn = NearestNeighbor(distance)
        nn.train(xt, mk.ndarray(listY))
    }

    private fun concatenate(array: Array<NDArray<Double, D2>>, axis: Int = 0): NDArray<Double, D2> {
        val ret = initMemoryView<Double>(array.size * array.first().size, DataType.DoubleDataType)
        fun MemoryView<Double>.add(index: Int, other: NDArray<Double, D2>) {
            var count = index
            for (element in other) {
                this[count++] = element
            }
        }

        var index = 0
        for (a in array) {
            ret.add(index, a)
            index += a.size
        }

        return NDArray(ret, 0, intArrayOf(array.size * array.first().shape.first(), array.first().shape[1]), dim = D2)
    }

    fun prepareTestSamples(min: Int = 0, max: Int = 2, step: Double = 0.01) {
        range = Pair(min, max)

        var y = 0.0
        var x = 0.0
        for (i in 0..40400) {
            xTest[i, 0] = y
            xTest[i, 1] = x
            x += step
            if (i > 0 && i % 200 == 0) {
                y += step
                x = 0.0
            }
        }
    }

    fun analyse() {
        yTest = nn.predict(xTest)

        for (label in 0 until nofClasses) {
            val classI = ArrayList<Double>()
            for (i in 0 until yTest.size) {
                if (yTest[i] == label) {
                    classI.addAll(xTest[i].toList())
                }
            }
            classified.add(mk.ndarray(classI))
        }
    }
}

fun main() {
    val x1 = generateRandomPoints(50, 0, 1)
    val x2 = generateRandomPoints(50, 1, 2)

    var tempX = generateRandomPoints(50, 0, 1)
    var listX = arrayListOf<List<Double>>()
    for (tx in 0 until tempX.shape.first()) {
        listX.add(arrayListOf(tempX[tx, 0], tempX[tx, 1] + 1))
    }
    val x3 = mk.ndarray(listX)

    tempX = generateRandomPoints(50, 1, 2)
    listX = arrayListOf()
    for (tx in 0 until tempX.shape.first()) {
        listX.add(arrayListOf(tempX[tx, 0], tempX[tx, 1] - 1))
    }
    val x4 = mk.ndarray(listX)

    val c4 = Analysis(arrayOf(x1, x2, x3, x4), distance = 1)
    c4.prepareTestSamples()
    c4.analyse()

    // accuracy
    var accuracy = 0.0
    for ((i, x_i) in (0 until c4.xTest.shape.first()).withIndex()) {
        val trueLabel: Int = if (c4.xTest[x_i, 0] < 1 && c4.xTest[x_i, 1] < 1) {
            0
        } else if (c4.xTest[x_i, 0] > 1 && c4.xTest[x_i, 1] > 1) {
            1
        } else if (c4.xTest[x_i, 0] < 1 && c4.xTest[x_i, 1] > 1) {
            2
        } else {
            3
        }
        if (trueLabel == c4.yTest[i]) {
            accuracy += 1
        }
    }
    accuracy /= c4.xTest.shape[0]
    println(accuracy)
}
