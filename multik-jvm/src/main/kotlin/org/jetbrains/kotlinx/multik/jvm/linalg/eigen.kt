package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.jvm.upperHessenberg
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.timesAssign
import java.lang.ArithmeticException
import java.lang.UnsupportedOperationException
import kotlin.math.*



public fun<T: Number> eig(a: MultiArray<T, D2>): MultiArray<ComplexDouble, D1> {
    requireSquare(a)
    val b = mk.d2arrayComplex(a.shape[0], a.shape[0]) { ComplexDouble(a[it / a.shape[0], it % a.shape[1]]) }
    return eigenvalues(b)
}

public fun<T: Complex> eigC(a: MultiArray<T, D2>): MultiArray<ComplexDouble, D1> {
    requireSquare(a)
    val b: MultiArray<ComplexDouble, D2>
    when (a.dtype) {
        DataType.ComplexFloatDataType -> {
            a as MultiArray<ComplexFloat, D2>
            b = mk.d2arrayComplex(a.shape[0], a.shape[0]) { ComplexDouble(a[it / a.shape[0], it % a.shape[1]].re, a[it / a.shape[0], it % a.shape[1]].im) }
        }
        DataType.ComplexDoubleDataType -> {
            a as MultiArray<ComplexDouble, D2>
            b = a
        }
        else -> {
            throw UnsupportedOperationException("matrix should be Complex")
        }
    }
    return eigenvalues(b)
}


/**
 * computes eigenvalues of matrix a
 */
internal fun eigenvalues(a: MultiArray<ComplexDouble, D2>): MultiArray<ComplexDouble, D1> {
    val (_, H) = upperHessenberg(a)
    val (upperTriangular, _) = qrShifted(H)

    return mk.d1arrayComplex(upperTriangular.shape[0]) { upperTriangular[it, it] }
}


/**
 * returns Schur decomposition:
 * Q, R matrices: Q * R * Q.H = a
 *
 * Q is unitary: Q * Q.H = Id
 * R is upper triangular
 * where mat.H is hermitian transpose of matrix mat
 */
internal fun schurDecomposition(a: MultiArray<ComplexDouble, D2>): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {
    val (L, H) = upperHessenberg(a)
    // a = L * H * L.H
    // H = L1 * UT * L1.H
    // a = (L * L1) * UT * (L1.H * L.H)
    val (upperTriangular, L1) = qrShifted(H)
    return Pair(tempDot(L, L1), upperTriangular) //TODO change to dot
}

/**
 * implementation of qr algorithm
 * matrix a must be in upper Hesenberg form
 */
private fun qrShifted(a: MultiArray<ComplexDouble, D2>, trialsNumber: Int = 30): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {

    val z = mk.identity<ComplexDouble>(a.shape[0])
    val v = mk.empty<ComplexDouble, D1>(2)

    val n = a.shape[0]
    val h = a.deepCopy() as D2Array<ComplexDouble>

    for (i in 1 until n) {
        if (h[i, i - 1].im != 0.0) {
            var sc = h[i, i - 1] / absL1(h[i, i - 1])
            sc = sc.conjugate() / sc.abs()
            h[i, i - 1] = h[i, i - 1].abs().toComplexDouble()
            (h[i, i..n] as D1Array<ComplexDouble>) *= sc
            (h[0..min(n, i + 2), i] as D1Array<ComplexDouble>) *= sc.conjugate()
            (z[0..n, i] as D1Array<ComplexDouble>) *= sc.conjugate()
        }
    }

    val safemin = 1e-300
    val ulp = 1e-16
    val smlnum = safemin * (n.toDouble() / ulp)

    val itmax = trialsNumber * max(10, n)
    var kdefl = 0

    var i = n - 1

    while (i >= 0) {

        var l = 0

        var failedToConverge = true

        for (iteration in 0 until itmax) { // Look for a single small subdiagonal element

            l = run {
                for (k in i downTo l + 1) {
                    if (absL1(h[k, k - 1]) < smlnum) {
                        return@run k
                    }

                    var tst = absL1(h[k - 1, k - 1]) + absL1(h[k, k])

                    if (tst == 0.0) {
                        if (k >= 2) {
                            tst += abs(h[k - 1, k - 2].re)
                        }
                        if (k + 2 <= n) {
                            tst += abs(h[k + 1, k].re)
                        }
                    }

                    if (abs(h[k, k - 1].re) <= ulp * tst) {
                        // The following is a conservative small subdiagonal
                        // deflation criterion due to Ahues & Tisseur (LAWN 122,
                        // 1997). It has better mathematical foundation and
                        // improves accuracy in some examples.
                        val ab = max(absL1(h[k, k - 1]), absL1(h[k - 1, k]))
                        val ba = min(absL1(h[k, k - 1]), absL1(h[k - 1, k]))
                        val aa = max(
                            absL1(h[k, k]),
                            absL1(h[k - 1, k - 1] - h[k, k])
                        )
                        val bb = min(
                            absL1(h[k, k]),
                            absL1(h[k - 1, k - 1] - h[k, k])
                        )

                        val s = aa + ab
                        if (ba * (ab / s) <= max(smlnum, ulp * (bb * (aa / s)))) {
                            return@run k
                        }
                    }
                }

                return@run l
            }

            if (l > 0) {
                h[l, l - 1] = ComplexDouble.zero
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
                    s = dat1 * abs(h[i, i - 1].re)
                    t = h[i, i] + s
                }
                kdefl % kExceptionalShift == 0 -> {
                    s = dat1 * abs(h[l + 1, l].re)
                    t = h[l, l] + s
                }
                else -> {
                    t = h[i, i]
                    u = csqrt(h[i - 1, i]) * csqrt(h[i, i - 1])
                    s = absL1(u)
                    if (s != 0.0) {
                        val x = (h[i - 1, i - 1] - t) * 0.5
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
                    h11 = h[m, m]
                    h22 = h[m + 1, m + 1]
                    h11s = h11 - t
                    h21 = h[m + 1, m].re.toComplexDouble()
                    s = absL1(h11s) + h21.abs()
                    h11s /= s
                    h21 /= s
                    v[0] = h11s
                    v[1] = h21
                    if (m > l) {
                        val h10 = h[m, m - 1]

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
                    v[0] = h[k, k - 1]
                    v[1] = h[k + 1, k - 1]
                }

                val (v1, t1) = computeHouseholderReflectorInplace(2, v[0], v[1..2] as D1Array<ComplexDouble>)
                v[0] = v1

                if (k > m) {
                    h[k, k - 1] = v[0]
                    h[k + 1, k - 1] = ComplexDouble.zero
                }

                val v2 = v[1]
                val t2 = (t1 * v2).re

                for (j in k until n) {
                    val sum = t1.conjugate() * h[k, j] + h[k + 1, j] * t2
                    h[k, j] -= sum
                    h[k + 1, j] = h[k + 1, j] - sum * v2
                }

                for (j in 0 until min(k + 2, i) + 1) {
                    val sum = t1 * h[j, k] + h[j, k + 1] * t2
                    h[j, k] -= sum
                    h[j, k + 1] -= sum * v2.conjugate()
                }

                for (j in 0 until n) {
                    val sum = t1 * z[j, k] + z[j, k + 1] * t2
                    z[j, k] -= sum
                    z[j, k + 1] -= sum * v2.conjugate()
                }

                if (k == m && m > l) {
                    var temp = ComplexDouble.one - t1
                    temp /= temp.abs()
                    h[m + 1, m] = h[m + 1, m] * temp.conjugate()
                    if (m + 2 < i) {
                        h[m + 2, m + 1] *= temp
                    }
                    for (j in m..i) {
                        if (j != m + 1) {
                            if (n > j + 1) {
                                (h[j, (j + 1)..n] as D1Array<ComplexDouble>) *= temp
                            }
                            h[0..j, j] as D1Array<ComplexDouble> *= temp.conjugate()
                            z[0..n, j] as D1Array<ComplexDouble> *= temp.conjugate()
                        }
                    }
                }
            }

            // Ensure that H(I,I-1) is real.
            var temp = h[i, i - 1]
            if (temp.im != 0.0) {
                val rtemp = temp.abs().toComplexDouble()
                h[i, i - 1] = rtemp
                temp /= rtemp
                if (n > i + 1) {
                    h[i, (i + 1)..n] as D1Array<ComplexDouble> *= temp.conjugate()
                }
                h[0..i, i] as D1Array<ComplexDouble> *= temp
                z[0..n, i] as D1Array<ComplexDouble> *= temp
            }
        }
        if (failedToConverge) {
            throw ArithmeticException("failed to converge with trialsNumber = $trialsNumber")
        }
        kdefl = 0
        i = l - 1
    }

    return Pair(h, z)
}

// return (beta, tau), mute x
private fun computeHouseholderReflectorInplace(
    n: Int,
    alpha: ComplexDouble,
    x: D1Array<ComplexDouble>
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

/** sign of number
 *
 * differs from builtin sign:
 * signum(0) = 1
 * sign(0) = 0
 */
fun signum(x: Double): Double = (if (x == 0.0) 1.0 else sign(x))
