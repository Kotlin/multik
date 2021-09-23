package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.d1arrayComplex
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.toComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.toComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.CopyStrategy
import org.jetbrains.kotlinx.multik.ndarray.operations.timesAssign
import org.jetbrains.kotlinx.multik.ndarray.operations.toType
import kotlin.collections.component1
import kotlin.collections.component2
import kotlin.math.*

internal fun <T, O : Any> eigenValuesCommon(a: MultiArray<T, D2>, dtype: DataType): D1Array<O> {
    requireSquare(a.shape)
    val mat = a.toType<T, O, D2>(dtype, CopyStrategy.MEANINGFUL)
    return when (dtype) {
        DataType.ComplexFloatDataType -> eigenvaluesFloat(mat as MultiArray<ComplexFloat, D2>)
        DataType.ComplexDoubleDataType -> eigenvaluesDouble(mat as MultiArray<ComplexDouble, D2>)
        else -> throw UnsupportedOperationException()
    } as D1Array<O>
}

/**
 * computes eigenvalues of matrix a
 */
internal fun eigenvaluesFloat(a: MultiArray<ComplexFloat, D2>): D1Array<ComplexFloat> {
    val (_, H) = upperHessenbergFloat(a)
    val (upperTriangular, _) = qrShiftedFloat(H)

    return mk.d1arrayComplex(upperTriangular.shape[0]) { upperTriangular[it, it] }
}

internal fun eigenvaluesDouble(a: MultiArray<ComplexDouble, D2>): D1Array<ComplexDouble> {
    val (_, H) = upperHessenbergDouble(a)
    val (upperTriangular, _) = qrShiftedDouble(H)

    return mk.d1arrayComplex(upperTriangular.shape[0]) { upperTriangular[it, it] }
}


/**
 * returns Schur decomposition:
 * Q, R matrices: Q * R * Q.H = a
 *
 * Q is unitary: Q * Q.H = Id
 * R is upper triangular
 * where mat.H is hermitian transpose of matrix mat
 *
 * NOTE: inplace
 */
internal fun schurDecomposition(a: MultiArray<ComplexDouble, D2>): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {
    val (L, H) = upperHessenbergDouble(a)
    // a = L * H * L.H
    // H = L1 * UT * L1.H
    // a = (L * L1) * UT * (L1.H * L.H)
    val (upperTriangular, L1) = qrShiftedDouble(H)
    return Pair(dotMatrixComplex(L, L1), upperTriangular)
}

/**
 * implementation of qr algorithm
 * matrix a must be in upper Hesenberg form
 *
 * NOTE: inplace function, change matrix a
 */
private fun qrShiftedFloat(
    a: MultiArray<ComplexFloat, D2>, trialsNumber: Int = 30
): Pair<D2Array<ComplexFloat>, D2Array<ComplexFloat>> {
    val (n, _) = a.shape
    val z = mk.identity<ComplexFloat>(a.shape[0])
    val v = mk.empty<ComplexFloat, D1>(2)

    a as D2Array<ComplexFloat>

    for (i in 1 until n) {
        if (a[i, i - 1].im != 0f) {
            var sc = a[i, i - 1] / absL1(a[i, i - 1])
            sc = sc.conjugate() / sc.abs()
            a[i, i - 1] = a[i, i - 1].abs().toComplexFloat()
            (a[i, i..n] as D1Array<ComplexFloat>) *= sc
            (a[0..min(n, i + 2), i] as D1Array<ComplexFloat>) *= sc.conjugate()
            (z[0..n, i] as D1Array<ComplexFloat>) *= sc.conjugate()
        }
    }

    val ulp = 1e-14f
    val smlnum = 1e-45f * (n / ulp)

    val itmax = trialsNumber * max(10, n)
    var kdefl = 0

    var i = n - 1

    while (i >= 0) {

        var l = 0

        var failedToConverge = true

        for (iteration in 0 until itmax) { // Look for a single small subdiagonal element

            l = run {
                for (k in i downTo l + 1) {
                    if (absL1(a[k, k - 1]) < smlnum) {
                        return@run k
                    }

                    var tst = absL1(a[k - 1, k - 1]) + absL1(a[k, k])

                    if (tst == 0f) {
                        if (k >= 2) {
                            tst += abs(a[k - 1, k - 2].re)
                        }
                        if (k + 2 <= n) {
                            tst += abs(a[k + 1, k].re)
                        }
                    }

                    if (abs(a[k, k - 1].re) <= ulp * tst) {
                        // The following is a conservative small subdiagonal
                        // deflation criterion due to Ahues & Tisseur (LAWN 122,
                        // 1997). It has better mathematical foundation and
                        // improves accuracy in some examples.
                        val ab = max(absL1(a[k, k - 1]), absL1(a[k - 1, k]))
                        val ba = min(absL1(a[k, k - 1]), absL1(a[k - 1, k]))
                        val aa = max(absL1(a[k, k]), absL1(a[k - 1, k - 1] - a[k, k]))
                        val bb = min(absL1(a[k, k]), absL1(a[k - 1, k - 1] - a[k, k]))

                        val s = aa + ab
                        if (ba * (ab / s) <= max(smlnum, ulp * (bb * (aa / s)))) {
                            return@run k
                        }
                    }
                }

                return@run l
            }

            if (l > 0) {
                a[l, l - 1] = ComplexFloat.zero
            }

            if (l >= i) {
                failedToConverge = false
                break
            }

            kdefl++

            val kExceptionalShift = 10
            val dat1 = 0.75f
            var s: Float
            var t: ComplexFloat
            var u: ComplexFloat

            when {
                kdefl % (2 * kExceptionalShift) == 0 -> {
                    s = dat1 * abs(a[i, i - 1].re)
                    t = a[i, i] + s
                }
                kdefl % kExceptionalShift == 0 -> {
                    s = dat1 * abs(a[l + 1, l].re)
                    t = a[l, l] + s
                }
                else -> {
                    t = a[i, i]
                    u = csqrt(a[i - 1, i]) * csqrt(a[i, i - 1])
                    s = absL1(u)
                    if (s != 0f) {
                        val x = (a[i - 1, i - 1] - t) * 0.5f
                        val sx = absL1(x)
                        s = max(s, absL1(x))
                        var y = csqrt((x / s) * (x / s) + (u / s) * (u / s)) * s
                        if (sx > 0.0 && ((x.re / sx) * y.re + (x.im / sx) * y.im) < 0.0) {
                            y = -y
                        }
                        t -= u * (u / (x + y))
                    }
                }
            }

            // Look for two consecutive small subdiagonal elements
            var h11: ComplexFloat
            var h11s: ComplexFloat
            var h22: ComplexFloat
            var h21: ComplexFloat

            val m = run {
                for (m in i - 1 downTo l) {
                    h11 = a[m, m]
                    h22 = a[m + 1, m + 1]
                    h11s = h11 - t
                    h21 = a[m + 1, m].re.toComplexFloat()
                    s = absL1(h11s) + h21.abs()
                    h11s /= s
                    h21 /= s
                    v[0] = h11s
                    v[1] = h21
                    if (m > l) {
                        val h10 = a[m, m - 1]

                        if (h10.abs() * h21.abs() <= ulp * (absL1(h11s) * (absL1(h11) + absL1(h22)))) {
                            return@run m
                        }
                    }
                }
                return@run l
            }

            // single-shift qr step
            for (k in m until i) {
                if (k > m) {
                    v[0] = a[k, k - 1]
                    v[1] = a[k + 1, k - 1]
                }

                val (v1, t1) = computeHouseholderReflectorInplace(2, v[0], v[1..2] as D1Array<ComplexFloat>)
                v[0] = v1

                if (k > m) {
                    a[k, k - 1] = v[0]
                    a[k + 1, k - 1] = ComplexFloat.zero
                }

                val v2 = v[1]
                val t2 = (t1 * v2).re

                for (j in k until n) {
                    val sum = t1.conjugate() * a[k, j] + a[k + 1, j] * t2
                    a[k, j] -= sum
                    a[k + 1, j] = a[k + 1, j] - sum * v2
                }

                for (j in 0 until min(k + 2, i) + 1) {
                    val sum = t1 * a[j, k] + a[j, k + 1] * t2
                    a[j, k] -= sum
                    a[j, k + 1] -= sum * v2.conjugate()
                }

                for (j in 0 until n) {
                    val sum = t1 * z[j, k] + z[j, k + 1] * t2
                    z[j, k] -= sum
                    z[j, k + 1] -= sum * v2.conjugate()
                }

                if (k == m && m > l) {
                    var temp = ComplexFloat.one - t1
                    temp /= temp.abs()
                    a[m + 1, m] = a[m + 1, m] * temp.conjugate()
                    if (m + 2 < i) {
                        a[m + 2, m + 1] *= temp
                    }
                    for (j in m..i) {
                        if (j != m + 1) {
                            if (n > j + 1) {
                                (a[j, (j + 1)..n] as D1Array<ComplexFloat>) *= temp
                            }
                            a[0..j, j] as D1Array<ComplexFloat> *= temp.conjugate()
                            z[0..n, j] as D1Array<ComplexFloat> *= temp.conjugate()
                        }
                    }
                }
            }

            // Ensure that h[i, i - 1] is real
            var temp = a[i, i - 1]
            if (temp.im != 0f) {
                val rtemp = temp.abs().toComplexFloat()
                a[i, i - 1] = rtemp
                temp /= rtemp
                if (n > i + 1) {
                    a[i, (i + 1)..n] as D1Array<ComplexFloat> *= temp.conjugate()
                }
                a[0..i, i] as D1Array<ComplexFloat> *= temp
                z[0..n, i] as D1Array<ComplexFloat> *= temp
            }
        }
        if (failedToConverge) {
            throw ArithmeticException("failed to converge with trialsNumber = $trialsNumber")
        }
        kdefl = 0
        i = l - 1
    }

    return Pair(a, z)
}

private fun qrShiftedDouble(
    a: MultiArray<ComplexDouble, D2>, trialsNumber: Int = 30
): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {
    val (n, _) = a.shape
    val z = mk.identity<ComplexDouble>(a.shape[0])
    val v = mk.empty<ComplexDouble, D1>(2)

    a as D2Array<ComplexDouble>

    for (i in 1 until n) {
        if (a[i, i - 1].im != 0.0) {
            var sc = a[i, i - 1] / absL1(a[i, i - 1])
            sc = sc.conjugate() / sc.abs()
            a[i, i - 1] = a[i, i - 1].abs().toComplexDouble()
            (a[i, i..n] as D1Array<ComplexDouble>) *= sc
            (a[0..min(n, i + 2), i] as D1Array<ComplexDouble>) *= sc.conjugate()
            (z[0..n, i] as D1Array<ComplexDouble>) *= sc.conjugate()
        }
    }

    val ulp = 1e-16
    val smlnum = (1e-300) * (n / ulp)

    val itmax = trialsNumber * max(10, n)
    var kdefl = 0

    var i = n - 1

    while (i >= 0) {

        var l = 0

        var failedToConverge = true

        for (iteration in 0 until itmax) { // Look for a single small subdiagonal element

            l = run {
                for (k in i downTo l + 1) {
                    if (absL1(a[k, k - 1]) < smlnum) {
                        return@run k
                    }

                    var tst = absL1(a[k - 1, k - 1]) + absL1(a[k, k])

                    if (tst == 0.0) {
                        if (k >= 2) {
                            tst += abs(a[k - 1, k - 2].re)
                        }
                        if (k + 2 <= n) {
                            tst += abs(a[k + 1, k].re)
                        }
                    }

                    if (abs(a[k, k - 1].re) <= ulp * tst) {
                        // The following is a conservative small subdiagonal
                        // deflation criterion due to Ahues & Tisseur (LAWN 122,
                        // 1997). It has better mathematical foundation and
                        // improves accuracy in some examples.
                        val ab = max(absL1(a[k, k - 1]), absL1(a[k - 1, k]))
                        val ba = min(absL1(a[k, k - 1]), absL1(a[k - 1, k]))
                        val aa = max(absL1(a[k, k]), absL1(a[k - 1, k - 1] - a[k, k]))
                        val bb = min(absL1(a[k, k]), absL1(a[k - 1, k - 1] - a[k, k]))

                        val s = aa + ab
                        if (ba * (ab / s) <= max(smlnum, ulp * (bb * (aa / s)))) {
                            return@run k
                        }
                    }
                }

                return@run l
            }

            if (l > 0) {
                a[l, l - 1] = ComplexDouble.zero
            }

            if (l >= i) {
                failedToConverge = false
                break
            }

            kdefl++

            val kExceptionalShift = 10
            val dat1 = 0.75
            var s: Double
            var t: ComplexDouble
            var u: ComplexDouble

            when {
                kdefl % (2 * kExceptionalShift) == 0 -> {
                    s = dat1 * abs(a[i, i - 1].re)
                    t = a[i, i] + s
                }
                kdefl % kExceptionalShift == 0 -> {
                    s = dat1 * abs(a[l + 1, l].re)
                    t = a[l, l] + s
                }
                else -> {
                    t = a[i, i]
                    u = csqrt(a[i - 1, i]) * csqrt(a[i, i - 1])
                    s = absL1(u)
                    if (s != 0.0) {
                        val x = (a[i - 1, i - 1] - t) * 0.5
                        val sx = absL1(x)
                        s = max(s, absL1(x))
                        var y = csqrt((x / s) * (x / s) + (u / s) * (u / s)) * s
                        if (sx > 0.0 && ((x.re / sx) * y.re + (x.im / sx) * y.im) < 0.0) {
                            y = -y
                        }
                        t -= u * (u / (x + y))
                    }
                }
            }

            // Look for two consecutive small subdiagonal elements
            var h11: ComplexDouble
            var h11s: ComplexDouble
            var h22: ComplexDouble
            var h21: ComplexDouble

            val m = run {
                for (m in i - 1 downTo l) {
                    h11 = a[m, m]
                    h22 = a[m + 1, m + 1]
                    h11s = h11 - t
                    h21 = a[m + 1, m].re.toComplexDouble()
                    s = absL1(h11s) + h21.abs()
                    h11s /= s
                    h21 /= s
                    v[0] = h11s
                    v[1] = h21
                    if (m > l) {
                        val h10 = a[m, m - 1]

                        if (h10.abs() * h21.abs() <= ulp * (absL1(h11s) * (absL1(h11) + absL1(h22)))) {
                            return@run m
                        }
                    }
                }
                return@run l
            }

            // single-shift qr step
            for (k in m until i) {
                if (k > m) {
                    v[0] = a[k, k - 1]
                    v[1] = a[k + 1, k - 1]
                }

                val (v1, t1) = computeHouseholderReflectorInplace(2, v[0], v[1..2] as D1Array<ComplexDouble>)
                v[0] = v1

                if (k > m) {
                    a[k, k - 1] = v[0]
                    a[k + 1, k - 1] = ComplexDouble.zero
                }

                val v2 = v[1]
                val t2 = (t1 * v2).re

                for (j in k until n) {
                    val sum = t1.conjugate() * a[k, j] + a[k + 1, j] * t2
                    a[k, j] -= sum
                    a[k + 1, j] = a[k + 1, j] - sum * v2
                }

                for (j in 0 until min(k + 2, i) + 1) {
                    val sum = t1 * a[j, k] + a[j, k + 1] * t2
                    a[j, k] -= sum
                    a[j, k + 1] -= sum * v2.conjugate()
                }

                for (j in 0 until n) {
                    val sum = t1 * z[j, k] + z[j, k + 1] * t2
                    z[j, k] -= sum
                    z[j, k + 1] -= sum * v2.conjugate()
                }

                if (k == m && m > l) {
                    var temp = ComplexFloat.one - t1
                    temp /= temp.abs()
                    a[m + 1, m] = a[m + 1, m] * temp.conjugate()
                    if (m + 2 < i) {
                        a[m + 2, m + 1] *= temp
                    }
                    for (j in m..i) {
                        if (j != m + 1) {
                            if (n > j + 1) {
                                (a[j, (j + 1)..n] as D1Array<ComplexDouble>) *= temp
                            }
                            a[0..j, j] as D1Array<ComplexDouble> *= temp.conjugate()
                            z[0..n, j] as D1Array<ComplexDouble> *= temp.conjugate()
                        }
                    }
                }
            }

            // Ensure that h[i, i - 1] is real
            var temp = a[i, i - 1]
            if (temp.im != 0.0) {
                val rtemp = temp.abs().toComplexDouble()
                a[i, i - 1] = rtemp
                temp /= rtemp
                if (n > i + 1) {
                    a[i, (i + 1)..n] as D1Array<ComplexDouble> *= temp.conjugate()
                }
                a[0..i, i] as D1Array<ComplexDouble> *= temp
                z[0..n, i] as D1Array<ComplexDouble> *= temp
            }
        }
        if (failedToConverge) {
            throw ArithmeticException("failed to converge with trialsNumber = $trialsNumber")
        }
        kdefl = 0
        i = l - 1
    }

    return Pair(a, z)
}

// return (beta, tau), mute x
private fun computeHouseholderReflectorInplace(
    n: Int, alpha: ComplexFloat, x: D1Array<ComplexFloat>
): Pair<ComplexFloat, ComplexFloat> {
    var tmp = alpha
    if (n <= 0) {
        return Pair(tmp, ComplexFloat.zero)
    }
    var xnorm = 0f
    for (i in 0 until n - 1) {
        xnorm += (x[i] * x[i].conjugate()).re
    }
    xnorm = sqrt(max(xnorm, 0f))
    var alphr = alpha.re
    var alphi = alpha.im

    if (xnorm == 0f && alphi == 0f) {
        return Pair(tmp, ComplexFloat.zero)
    }

    var beta = -signum(alphr) * sqrt(alphr * alphr + alphi * alphi + xnorm * xnorm)
    val safemin = 2e-45f
    val rsafemin = 1f / safemin

    var knt = 0
    while (abs(beta) < safemin) {
        knt++
        x[0..(n - 1)] as D1Array<ComplexFloat> *= rsafemin.toComplexFloat()
        beta *= rsafemin
        alphi *= rsafemin
        alphr *= safemin

        if (abs(beta) < safemin && knt < 20) {
            continue
        }

        xnorm = 0f
        for (i in 0 until n - 1) {
            xnorm += (x[i] * x[i].conjugate()).re
        }
        xnorm = sqrt(max(xnorm, 0f))
        tmp = ComplexFloat(alphr, alphi)
        beta = -signum(alphr) * sqrt(alphr * alphr + alphi * alphi + xnorm * xnorm)
    }

    val tau = ComplexFloat((beta - alphr) / beta, -alphi / beta)
    tmp = 1.0.toComplexFloat() / (tmp - beta)

    x[0..(n - 1)] as D1Array<ComplexFloat> *= tmp
    for (j in 1..knt) {
        beta *= safemin
    }


    return Pair(beta.toComplexFloat(), tau)
}

private fun computeHouseholderReflectorInplace(
    n: Int, alpha: ComplexDouble, x: D1Array<ComplexDouble>
): Pair<ComplexDouble, ComplexDouble> {
    var tmp = alpha
    if (n <= 0) {
        return Pair(tmp, ComplexDouble.zero)
    }
    var xnorm = 0.0
    for (i in 0 until n - 1) {
        xnorm += (x[i] * x[i].conjugate()).re
    }
    xnorm = sqrt(max(xnorm, 0.0))
    var alphr = alpha.re
    var alphi = alpha.im

    if (xnorm == 0.0 && alphi == 0.0) {
        return Pair(tmp, ComplexDouble.zero)
    }

    var beta = -signum(alphr) * sqrt(alphr * alphr + alphi * alphi + xnorm * xnorm)
    val safemin = 2e-300
    val rsafemin = 1.0 / safemin

    var knt = 0
    while (abs(beta) < safemin) {
        knt++
        x[0..(n - 1)] as D1Array<ComplexDouble> *= rsafemin.toComplexDouble()
        beta *= rsafemin
        alphi *= rsafemin
        alphr *= safemin

        if (abs(beta) < safemin && knt < 20) {
            continue
        }

        xnorm = 0.0
        for (i in 0 until n - 1) {
            xnorm += (x[i] * x[i].conjugate()).re
        }
        xnorm = sqrt(max(xnorm, 0.0))
        tmp = ComplexDouble(alphr, alphi)
        beta = -signum(alphr) * sqrt(alphr * alphr + alphi * alphi + xnorm * xnorm)
    }

    val tau = ComplexDouble((beta - alphr) / beta, -alphi / beta)
    tmp = 1.0.toComplexDouble() / (tmp - beta)

    x[0..(n - 1)] as D1Array<ComplexDouble> *= tmp
    for (j in 1..knt) {
        beta *= safemin
    }


    return Pair(beta.toComplexDouble(), tau)
}

// complex number L1 norm
private fun absL1(a: ComplexDouble): Double = abs(a.re) + abs(a.im)

private fun absL1(a: ComplexFloat): Float = abs(a.re) + abs(a.im)

/** sign of number
 *
 * differs from builtin sign:
 * signum(0) = 1
 * sign(0) = 0
 */
fun signum(x: Float): Float = (if (x == 0f) 1f else sign(x))

fun signum(x: Double): Double = (if (x == 0.0) 1.0 else sign(x))
