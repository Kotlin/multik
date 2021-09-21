package org.jetbrains.kotlinx.multik.examples

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import kotlin.math.abs
import kotlin.random.Random


//operator fun <T : Number, D : Dimension> T.div(a: NDArray<T, D>): NDArray<T, D> {
//    val data = initMemoryView<T>(a.size, a.dtype)
//    val iterRight = a.iterator()
//    for (i in a.indices) {
//        data[i] = this / iterRight.next()
//    }
//    return NDArray(data, shape = a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
//}

//operator fun <T : Number, D : Dimension> T.times(a: NDArray<T, D>): NDArray<T, D> {
//    val data = initMemoryView<T>(a.size, a.dtype)
//    val iterRight = a.iterator()
//    for (i in a.indices) {
//        data[i] = this * iterRight.next()
//    }
//    return NDArray(data, shape = a.shape.copyOf(), dtype = a.dtype, dim = a.dim)
//}

fun sig(x: NDArray<Double, D2>, deriv: Boolean = false): NDArray<Double, D2> {
    //todo (add unary minus, add primitive arith)
    if (deriv) return x * (x * (-1.0) + 1.0)
    return 1.0 / (mk.math.exp(x * -1.0) + 1.0)
}

fun simpleNN() {
    val x = mk.ndarray(mk[mk[0.0, 0.0, 1.0], mk[0.0, 1.0, 1.0], mk[1.0, 0.0, 1.0], mk[1.0, 1.0, 1.0]])
    val y = mk.ndarray(mk[mk[0.0, 1.0, 1.0, 0.0]]).transpose()

    //todo (add random)
    val random = Random(3)
    val syn0 = 2.0 * mk.d2array(3, 4) { random.nextDouble() } - 1.0
    val syn1 = 2.0 * mk.d2array(4, 1) { random.nextDouble() } - 1.0

    var l0: NDArray<Double, D2>
    var l1: NDArray<Double, D2>
    var l2: NDArray<Double, D2>? = null

    for (i in 0..60_000) {
        l0 = x
        l1 = sig(mk.linalg.dot(l0, syn0))
        l2 = sig(mk.linalg.dot(l1, syn1))

        val l2Error = y - l2
        //todo (mean and absolute)
        if ((i % 10_000) == 0) {
            var sum = 0.0
            for (el in l2Error) {
                sum += abs(el)
            }
            println("step: $i, error: ${sum / l2Error.size}")
        }

        val l2Delta = l2Error * sig(l2, true)

        val l1Error = mk.linalg.dot(l2Delta, syn1.transpose())
        val l1Delta = l1Error * sig(l1, true)

        syn1 += mk.linalg.dot(l1.transpose(), l2Delta)
        syn0 += mk.linalg.dot(l0.transpose(), l1Delta)
    }
    println("Data: \n$l2")
}

fun main() {
    simpleNN()
}