package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.toComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.toComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.set
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sqrt


fun qrDouble(mat: MultiArray<Double, D2>): Pair<D2Array<Double>, D2Array<Double>> {
    var q = mk.identity<Double>(mat.shape[0]) // TODO shape q = m x n
    val r = mat.deepCopy() as D2Array<Double> // TODO shape r = min(m, n) x n
    for (i in 0 until min(mat.shape[0], mat.shape[1])) {
        val (tau, v) = householderTransformDouble(r[i..r.shape[0], i..r.shape[1]] as D2Array<Double>)
        val appliedR = applyHouseholderDouble(r[i..r.shape[0], i..r.shape[1]] as D2Array<Double>, tau, v)
        for (d0 in i until r.shape[0]) {
            for (d1 in i until r.shape[1]) {
                r[d0, d1] = appliedR[d0 - i, d1 - i]
            }
        }
        q = q.transpose()
        val appliedQ = applyHouseholderDouble(q[i..q.shape[0], 0..q.shape[1]] as D2Array<Double>, tau, v)
        for (d0 in i until q.shape[0]) {
            for (d1 in 0 until q.shape[1]) {
                q[d0, d1] = appliedQ[d0 - i, d1]
            }
        }
        q = q.transpose()
    }
    return Pair(q, r)
}

fun qrFloat(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>> {
    var q = mk.identity<Float>(mat.shape[0])
    val r = mat.deepCopy() as D2Array<Float>
    for (i in 0 until min(mat.shape[0], mat.shape[1])) {
        val (tau, v) = householderTransformFloat(r[i..r.shape[0], i..r.shape[1]] as D2Array<Float>)
        val appliedR = applyHouseholderFloat(r[i..r.shape[0], i..r.shape[1]] as D2Array<Float>, tau, v)
        for (d0 in i until r.shape[0]) {
            for (d1 in i until r.shape[1]) {
                r[d0, d1] = appliedR[d0 - i, d1 - i]
            }
        }
        q = q.transpose()
        val appliedQ = applyHouseholderFloat(q[i..q.shape[0], 0..q.shape[1]] as D2Array<Float>, tau, v)
        for (d0 in i until q.shape[0]) {
            for (d1 in 0 until q.shape[1]) {
                q[d0, d1] = appliedQ[d0 - i, d1]
            }
        }
        q = q.transpose()
    }
    return Pair(q, r)
}

fun qrComplexFloat(mat: MultiArray<ComplexFloat, D2>): Pair<D2Array<ComplexFloat>, D2Array<ComplexFloat>> {
    var q = mk.identity<ComplexFloat>(mat.shape[0])
    val r = mat.deepCopy() as D2Array<ComplexFloat>

    for (i in 0 until min(mat.shape[0], mat.shape[1])) {
        val (tau, v) = householderTransformComplexFloat(r[i..r.shape[0], i..r.shape[1]])
        val appliedR = applyHouseholderComplexFloat(r[i..r.shape[0], i..r.shape[1]] as D2Array<ComplexFloat>, tau, v)

        for (d0 in i until r.shape[0]) {
            for (d1 in i until r.shape[1]) {
                r[d0, d1] = appliedR[d0 - i, d1 - i]
            }
        }
        q = q.transpose()
        for (i1 in 0 until q.shape[0]) {
            for (j1 in 0 until q.shape[1]) {
                q[i1, j1] = q[i1, j1].conjugate()
            }
        }

        val appliedQ = applyHouseholderComplexFloat(q[i..q.shape[0], 0..q.shape[1]] as D2Array<ComplexFloat>, tau, v)
        for (d0 in i until q.shape[0]) {
            for (d1 in 0 until q.shape[1]) {
                q[d0, d1] = appliedQ[d0 - i, d1]
            }
        }
        q = q.transpose()
        for (i1 in 0 until q.shape[0]) {
            for (j1 in 0 until q.shape[1]) {
                q[i1, j1] = q[i1, j1].conjugate()
            }
        }
    }

    for (i in 1 until q.shape[0]) {
        for (j in 0 until i) {
            r[i, j] = ComplexFloat.zero
        }
    }

    return Pair(q, r)
}

fun qrComplexDouble(mat: MultiArray<ComplexDouble, D2>): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {

    var q = mk.identity<ComplexDouble>(mat.shape[0])
    val r = mat.deepCopy() as D2Array<ComplexDouble>
    for (i in 0 until min(mat.shape[0], mat.shape[1])) {
        val (tau, v) = householderTransformComplexDouble(r[i..r.shape[0], i..r.shape[1]] as D2Array<ComplexDouble>)
        val appliedR = applyHouseholderComplexDouble(r[i..r.shape[0], i..r.shape[1]] as D2Array<ComplexDouble>, tau, v)
        for (d0 in i until r.shape[0]) {
            for (d1 in i until r.shape[1]) {
                r[d0, d1] = appliedR[d0 - i, d1 - i]
            }
        }
        q = q.transpose()
        for (i1 in 0 until q.shape[0]) {
            for (j1 in 0 until q.shape[1]) {
                q[i1, j1] = q[i1, j1].conjugate()
            }
        }

        val appliedQ = applyHouseholderComplexDouble(q[i..q.shape[0], 0..q.shape[1]] as D2Array<ComplexDouble>, tau, v)
        for (d0 in i until q.shape[0]) {
            for (d1 in 0 until q.shape[1]) {
                q[d0, d1] = appliedQ[d0 - i, d1]
            }
        }
        q = q.transpose()
        for (i1 in 0 until q.shape[0]) {
            for (j1 in 0 until q.shape[1]) {
                q[i1, j1] = q[i1, j1].conjugate()
            }
        }
    }

    for (i in 1 until q.shape[0]) {
        for (j in 0 until i) {
            r[i, j] = ComplexDouble.zero
        }
    }

    return Pair(q, r)
}

internal fun householderTransformDouble(x: D2Array<Double>): Pair<Double, D1Array<Double>> {
    val alpha = x[0, 0]

    var xnorm = 0.0
    for (i in 1 until x.shape[0]) {
        xnorm += x[i, 0] * x[i, 0]
    }
    xnorm = sqrt(xnorm)

    val v: D1Array<Double> = mk.empty<Double, D1>(x.shape[0])
    v[0] = 1.0
    if (xnorm == 0.0) {
        return Pair(0.0, v)
    }
    val beta = -(if (alpha >= 0) 1 else -1) * hypot(alpha, xnorm)
    val tau = (beta - alpha) / beta
    val alphaMinusBeta = alpha - beta
    for (i in 1 until v.size) {
        v[i] = x[i, 0] / alphaMinusBeta
    }
    return Pair(tau, v)
}

internal fun householderTransformFloat(x: D2Array<Float>): Pair<Float, D1Array<Float>> {
    val alpha = x[0, 0]

    var xnorm = 0f
    for (i in 1 until x.shape[0]) {
        xnorm += x[i, 0] * x[i, 0]
    }
    xnorm = sqrt(xnorm)

    val v = mk.empty<Float, D1>(x.shape[0])
    v[0] = 1f
    if (xnorm == 0f) {
        return Pair(0f, v)
    }
    val beta = -(if (alpha >= 0) 1 else -1) * hypot(alpha, xnorm)
    val tau = (beta - alpha) / beta
    val alphaMinusBeta = alpha - beta
    for (i in 1 until v.size) {
        v[i] = x[i, 0] / alphaMinusBeta
    }
    return Pair(tau, v)
}

internal fun householderTransformComplexFloat(x: MultiArray<ComplexFloat, D2>): Pair<ComplexFloat, D1Array<ComplexFloat>> {
    val alpha = x[0, 0]
//    val xnorm: Double = sqrt(x[1..x.shape[0], 0].map { it * it }.sum())
    var xnorm = 0f
    for (i in 1 until x.shape[0]) {
        xnorm += (x[i, 0] * x[i, 0].conjugate()).abs()
    }
    xnorm = sqrt(xnorm)

    val v = mk.empty<ComplexFloat, D1>(x.shape[0])
    v[0] = ComplexFloat.one
    if (xnorm == 0f) {
        return Pair(ComplexFloat.zero, v)
    }
    val beta = -(if (alpha.re >= 0) 1 else -1) * sqrt(alpha.re * alpha.re + alpha.im * alpha.im + xnorm * xnorm)
    val tau = (beta.toComplexFloat() - alpha) / beta.toComplexFloat()
    val coeff = ComplexFloat.one / (alpha - beta)
    for (i in 1 until v.size) {
        v[i] = x[i, 0] * coeff
    }
    return Pair(tau.conjugate(), v)
}

internal fun householderTransformComplexDouble(x: MultiArray<ComplexDouble, D2>): Pair<ComplexDouble, D1Array<ComplexDouble>> {
    val alpha = x[0, 0]
//    val xnorm: Double = sqrt(x[1..x.shape[0], 0].map { it * it }.sum())
    var xnorm = 0.0
    for (i in 1 until x.shape[0]) {
        xnorm += (x[i, 0] * x[i, 0].conjugate()).abs()
    }
    xnorm = sqrt(xnorm)

    val v: D1Array<ComplexDouble> = mk.empty<ComplexDouble, D1>(x.shape[0])
    v[0] = ComplexDouble.one
    if (xnorm == 0.0) {
        return Pair(ComplexDouble.zero, v)
    }
    val beta = -(if (alpha.re >= 0) 1 else -1) * sqrt(alpha.re * alpha.re + alpha.im * alpha.im + xnorm * xnorm)
    val tau = (beta.toComplexDouble() - alpha) / beta.toComplexDouble()
    val coeff = ComplexDouble.one / (alpha - beta)
    for (i in 1 until v.size) {
        v[i] = x[i, 0] * coeff
    }
    return Pair(tau.conjugate(), v)
}

internal fun applyHouseholderDouble(x: D2Array<Double>, tau: Double, v: D1Array<Double>): D2Array<Double> {
    //x - tau * np.sum(v * x) * v
    val applied = x.deepCopy()

    for (columnNumber in 0 until x.shape[1]) {
        var scal = 0.0 // scal(x[:, columnNumber], v)
        for (i in 0 until v.size) {
            scal += v[i] * x[i, columnNumber]
        }
        for (i in 0 until v.size) {
            applied[i, columnNumber] -= tau * scal * v[i]
        }
    }
    return applied
}

internal fun applyHouseholderFloat(x: D2Array<Float>, tau: Float, v: D1Array<Float>): D2Array<Float> {
    //x - tau * np.sum(v * x) * v
    val applied = x.deepCopy()

    for (columnNumber in 0 until x.shape[1]) {
        var scal = 0f // scal(x[:, columnNumber], v)
        for (i in 0 until v.size) {
            scal += v[i] * x[i, columnNumber]
        }
        for (i in 0 until v.size) {
            applied[i, columnNumber] -= tau * scal * v[i]
        }
    }
    return applied
}

internal fun applyHouseholderComplexFloat(
    x: MultiArray<ComplexFloat, D2>, tau: ComplexFloat, v: MultiArray<ComplexFloat, D1>
): D2Array<ComplexFloat> {
    //x - tau * np.sum(v * x) * v
    val applied = x.deepCopy() as D2Array<ComplexFloat>

    for (columnNumber in 0 until x.shape[1]) {
        var scal = ComplexFloat.zero // scal(x[:, columnNumber], v)
        for (i in 0 until v.size) {
            scal += v[i].conjugate() * x[i, columnNumber]
        }
        for (i in 0 until v.size) {
            applied[i, columnNumber] -= tau * scal * v[i]
        }
    }
    return applied
}

// returns (Id - tau * v * v.H) * x
// v.H is Hermitian conjugation
internal fun applyHouseholderComplexDouble(
    x: MultiArray<ComplexDouble, D2>, tau: ComplexDouble, v: MultiArray<ComplexDouble, D1>
): D2Array<ComplexDouble> {
    //x - tau * np.sum(v * x) * v
    val applied = x.deepCopy() as D2Array<ComplexDouble>

    for (columnNumber in 0 until x.shape[1]) {
        var scal = ComplexDouble.zero // scal(x[:, columnNumber], v)
        for (i in 0 until v.size) {
            scal += v[i].conjugate() * x[i, columnNumber]
        }
        for (i in 0 until v.size) {
            applied[i, columnNumber] -= tau * scal * v[i]
        }
    }
    return applied
}
