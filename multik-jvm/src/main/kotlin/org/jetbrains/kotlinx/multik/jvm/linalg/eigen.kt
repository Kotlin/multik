package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.jvm.linalg.csqrt
import org.jetbrains.kotlinx.multik.jvm.linalg.tempDot
import org.jetbrains.kotlinx.multik.jvm.linalg.toComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.timesAssign
import java.lang.ArithmeticException
import kotlin.math.*

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
fun qrShifted(a: MultiArray<ComplexDouble, D2>, trialsNumber: Int = 30): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {

    val w = mk.empty<ComplexDouble, D1>(a.shape[0])
    val z = mk.identity<ComplexDouble>(a.shape[0])
    val v = mk.empty<ComplexDouble, D1>(2)

    val dat1 = 0.75
    val kExceptionalShift = 10
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
    val ulp = 1e-16 // precision // todo: look
    val smlnum = safemin * (n.toDouble() / ulp)

    val itmax = trialsNumber * max(10, n)
    var kdefl = 0

    var i = n

    while (i >= 1) {

        var l = 1
        var failedToConverge = true

        for (iteration in 0 until itmax) { // Look for a single small subdiagonal element

            l = run {
                if (i < l + 1) {
                    return@run i
                }
                for (k in i downTo l + 1) {
                    if (absL1(h[k - 1, k - 2]) < smlnum) {
                        return@run k
                    }

                    var tst = absL1(h[k - 2, k - 2]) + absL1(h[k - 1, k - 1])

                    if (tst == 0.0) {
                        if (k - 2 >= 1) {
                            tst += abs(h[k - 2, k - 3].re)
                        }
                        if (k + 1 <= n) {
                            tst += abs(h[k, k - 1].re)
                        }
                    }

                    if (abs(h[k - 1, k - 2].re) <= ulp * tst) {

                        // ==== The following is a conservative small subdiagonal
                        // *           .    deflation criterion due to Ahues & Tisseur (LAWN 122,
                        // *           .    1997). It has better mathematical foundation and
                        // *           .    improves accuracy in some examples.  ====
                        val ab = max(absL1(h[k - 1, k - 2]), absL1(h[k - 2, k - 1]))
                        val ba = min(absL1(h[k - 1, k - 2]), absL1(h[k - 2, k - 1]))
                        val aa = max(
                            absL1(h[k - 1, k - 1]),
                            absL1(h[k - 2, k - 2] - h[k - 1, k - 1])
                        )
                        val bb = min(
                            absL1(h[k - 1, k - 1]),
                            absL1(h[k - 2, k - 2] - h[k - 1, k - 1])
                        )

                        val s = aa + ab
                        if (ba * (ab / s) <= max(smlnum, ulp * (bb * (aa / s)))) {
                            return@run k
                        }
                    }
                }

                return@run l
            }

            if (l > 1) {
                h[l - 1, l - 2] = ComplexDouble.zero
            }

            if (l >= i) {
                failedToConverge = false
                break
            }

            kdefl++

            var s: Double
            var t: ComplexDouble
            var u: ComplexDouble

            when {
                kdefl % (2 * kExceptionalShift) == 0 -> {
                    s = dat1 * abs(h[i - 1, i - 2].re)
                    t = s.toComplexDouble() + h[i - 1, i - 1]
                }
                kdefl % kExceptionalShift == 0 -> {
                    s = dat1 * abs(h[l, l - 1].re)
                    t = s.toComplexDouble() + h[l - 1, l - 1]
                }
                else -> {
                    t = h[i - 1, i - 1]
                    u = csqrt(h[i - 2, i - 1]) * csqrt(h[i - 1, i - 2])
                    s = absL1(u)
                    if (s != 0.0) {
                        val x = 0.5.toComplexDouble() * (h[i - 2, i - 2] - t)
                        val sx = absL1(x)
                        s = max(s, absL1(x))
                        var y = s.toComplexDouble() * csqrt((x / s) * (x / s) + (u / s) * (u / s))
                        if (sx > 0.0 && (x / sx).re * y.re + (x / sx).im * y.im < 0.0) {
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
                for (m in (i - 1) downTo (l + 1)) {
                    h11 = h[m - 1, m - 1]
                    h22 = h[m, m + 1 - 1]
                    h11s = h11 - t
                    h21 = h[m, m - 1].re.toComplexDouble()
                    s = absL1(h11s) + h21.abs()
                    h11s /= s
                    h21 /= s
                    v[0] = h11s
                    v[1] = h21
                    val h10 = h[m - 1, m - 2]

                    if (h10.abs() * h21.abs() <= ulp * (absL1(h11s) * (absL1(h11) + absL1(h22)))) {
                        return@run m
                    }
                }

                h11 = h[l - 1, l - 1]
                h22 = h[l, l]
                h11s = h11 - t
                h21 = h[l, l - 1].re.toComplexDouble()

                s = absL1(h11s) + h21.abs()
                h11s /= s
                h21 /= s
                v[0] = h11s
                v[1] = h21
                return@run l
            }

            // single-shift qr step
            for (k in m until i) {
                if (k > m) {
                    v[0] = h[k - 1, k - 2]
                    v[1] = h[k, k - 2]
                }

                // zlarfg( 2, v( 1 ), v( 2 ), 1, t1 )
                val (v1, t1) = computeHouseholderReflectorInline(2, v[0], v[1..2] as D1Array<ComplexDouble>)
                v[0] = v1

                if (k > m) {
                    h[k - 1, k - 2] = v[0]
                    h[k, k - 2] = 0.0.toComplexDouble()
                }

                val v2 = v[1]
                val t2 = (t1 * v2).re

                for (j in k..n) {
                    val sum = t1.conjugate() * h[k - 1, j - 1] + t2.toComplexDouble() * h[k, j - 1]
                    h[k - 1, j - 1] = h[k - 1, j - 1] - sum
                    h[k, j - 1] = h[k, j - 1] - sum * v2
                }

                for (j in 1..min(k + 2, i)) {
                    val sum = t1 * h[j - 1, k - 1] + t2.toComplexDouble() * h[j - 1, k]
                    h[j - 1, k - 1] -= sum
                    h[j - 1, k] -= sum * v2.conjugate()
                }

                for (j in 1..n) {
                    val sum = t1 * z[j - 1, k - 1] + t2.toComplexDouble() * z[j - 1, k]
                    z[j - 1, k - 1] -= sum
                    z[j - 1, k] -= sum * v2.conjugate()
                }

                if (k == m && m > l) {
                    var temp = 1.0.toComplexDouble() - t1
                    temp /= temp.abs()
                    h[m, m - 1] = h[m, m - 1] * temp.conjugate()
                    if (m + 2 < i) {
                        h[m + 1, m] *= temp
                    }
                    for (j in m..i) {
                        if (j != m + 1) {
                            if (n > j) {
                                (h[j - 1, j..n] as D1Array<ComplexDouble>) *= temp
                            }
                            h[0..(j - 1), j - 1] as D1Array<ComplexDouble> *= temp.conjugate()
                            z[0..n, j - 1] as D1Array<ComplexDouble> *= temp.conjugate()
                        }
                    }
                }
            }

            // Ensure that H(I,I-1) is real.
            var temp = h[i - 1, i - 2]
            if (temp.im != 0.0) {
                val rtemp = temp.abs().toComplexDouble()
                h[i - 1, i - 2] = rtemp
                temp /= rtemp
                if (n > i) {
                    h[i - 1, i..n] as D1Array<ComplexDouble> *= temp.conjugate()
                }
                h[0..(i - 1), i - 1] as D1Array<ComplexDouble> *= temp;
                z[0..n, i - 1] as D1Array<ComplexDouble> *= temp
            }
        }
        if (failedToConverge) {
            throw ArithmeticException("failed to converge, try to increase trialsNumber")
        }
        w[i - 1] = h[i - 1, i - 1]
        kdefl = 0
        i = l - 1
    }

    return Pair(h, z)
}

// return (beta, tau), mute x
fun computeHouseholderReflectorInline(
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
