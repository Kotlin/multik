package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.jvm.linalg.csqrt
import org.jetbrains.kotlinx.multik.jvm.linalg.tempDot
import org.jetbrains.kotlinx.multik.jvm.linalg.toComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.mapMultiIndexed
import kotlin.math.*

/**
 * computes eigenvalues of matrix a
 */
internal fun eigenvalues(a: MultiArray<ComplexDouble, D2>): MultiArray<ComplexDouble, D1> {
    val (L, H) = upperHessenberg((a as NDArray<ComplexDouble, D2>))
    val (upperTriangular, _) = qrShifted(H)

    val diagonal = mk.empty<ComplexDouble, D1>(upperTriangular.shape[0])
    for (i in 0 until upperTriangular.shape[0]) {
        diagonal[i] = upperTriangular[i, i]
    }
    return diagonal

}


/**
 * returns Schur decomplosition:
 * Q, R matrices: Q * R * Q.H = a
 *
 * Q is unitary: Q * Q.H = Id
 * R is upper triangular
 * where _.H is hermitian transpose
 */
internal fun schurDecomposition(a: MultiArray<ComplexDouble, D2>): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {
    val (L, H) = upperHessenberg((a as NDArray<ComplexDouble, D2>))
    val (upperTriangular, L1) = qrShifted(H)
    return Pair(tempDot(L, L1), upperTriangular)
}

/** implementation of qr algorithm
 *
 * matrix a must be in upper Hesenberg form
 */
fun qrShifted(a: MultiArray<ComplexDouble, D2>): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {

    val TRANS = -1

    val w = mk.empty<ComplexDouble, D1>(a.shape[0])
    val z = mk.empty<ComplexDouble, D2>(a.shape[0], a.shape[0])
    for (i in 0 until z.shape[0]) {
        z[i, i] = 1.0.toComplexDouble()
    }
    val v = mk.empty<ComplexDouble, D1>(2)
    val dat1 = 3.0 / 4.0
    val kexsh = 10
    val n = a.shape[0]
    val h = a.deepCopy() as D2Array<ComplexDouble>
    val jlo = 1
    val jhi = n
    val ilo = 1
    val ihi = n
    val iloz = 1
    val ihiz = n
    val ldz = n


    for (i in (ilo + 1)..ihi) {
        if (h[i + TRANS, i - 1 + TRANS].im != 0.0) {
            var sc = h[i + TRANS, i - 1 + TRANS] / cabs1(h[i + TRANS, i - 1 + TRANS])
            sc = sc.conjugate() / sc.abs()
            h[i + TRANS, i - 1 + TRANS] = h[i + TRANS, i - 1 + TRANS].abs().toComplexDouble()
            for (ii in 0 until (jhi - i + 1)) {
                h[i + TRANS, i + ii + TRANS] *= sc
            }
            for (ii in 0 until (min(jhi, i + 1) - jlo + 1)) {
                h[jlo + ii + TRANS, i + TRANS] *= sc.conjugate()
            }
            for (ii in 0 until ihiz - iloz + 1) {
                z[iloz + ii + TRANS, i + TRANS] *= sc.conjugate()
            }
        }
    }

    val nh = ihi - ilo + 1
    val nz = ihiz - iloz + 1


    val safemin = 1e-300
    val safemax = 1.0 / safemin
    val ulp = 1e-16 // precision
    val smlnum = safemin * (nh.toDouble() / ulp)


    val i1 = 1
    val i2 = n
    val itmax = 30 * max( 10, nh )
    var kdefl = 0


    var i = ihi
    while(true) {
        if (i < ilo) {
            break
        }
        var l = ilo


        for (its in 0..itmax) { // Look for a single small subdiagonal element
            if(its == itmax) {
                break
            }

            var foundk = l
            for (k in i downTo l + 1) {
                foundk = k
                if (cabs1(h[k + TRANS, k - 1 + TRANS]) < smlnum) {
                    break
                }
                var tst = cabs1(h[k - 1 + TRANS, k - 1 + TRANS]) + cabs1(h[k + TRANS, k + TRANS])
                if (tst == 0.0) {
                    if (k - 2 >= ilo) {
                        tst += abs(h[k - 1 + TRANS, k - 2 + TRANS].re)
                    }
                    if (k + 1 <= ihi) {
                        tst += abs(h[k + 1 + TRANS, k + TRANS].re)
                    }
                }
                if (abs(h[k + TRANS, k - 1 + TRANS].re) <= ulp * tst) {
                    val ab = max(cabs1(h[k + TRANS, k - 1 + TRANS] ), cabs1(h[k - 1 + TRANS, k + TRANS]))
                    val ba = min(cabs1(h[k + TRANS, k - 1 + TRANS]), cabs1(h[k - 1 + TRANS, k + TRANS]))
                    val aa = max(cabs1(h[k + TRANS, k + TRANS]), cabs1(h[k - 1 + TRANS, k - 1 + TRANS] - h[k + TRANS, k + TRANS]))
                    val bb = min(cabs1(h[k + TRANS, k + TRANS]), cabs1(h[k - 1 + TRANS, k - 1 + TRANS] - h[k + TRANS, k + TRANS]))
                    val s = aa + ab
                    if (ba * (ab / s) <=
                        max(smlnum, ulp * (bb * (aa / s)))) {
                        break
                    }
                }
                foundk = k - 1
            }
            l = foundk

            if (l > ilo) {
                h[l + TRANS, l - 1 + TRANS] = ComplexDouble.zero
            }
            if (l >= i) {
                break
            }

            kdefl++

            var s = 0.0
            var t = ComplexDouble.zero
            var u = ComplexDouble.zero

            when {
                kdefl % (2 * kexsh) == 0 -> {
                    s = dat1*abs(h[i + TRANS, i - 1 + TRANS].re)
                    t = s.toComplexDouble() + h[i + TRANS, i + TRANS]
                }
                kdefl % kexsh == 0 -> {
                    s = dat1 * abs(h[l + 1 + TRANS, l + TRANS].re)
                    t = s.toComplexDouble() + h[l + TRANS, l + TRANS]
                }
                else -> {
                    t = h[i + TRANS, i + TRANS]
                    u = csqrt(h[i - 1 + TRANS, i + TRANS]) * csqrt(h[i + TRANS, i - 1 + TRANS])
                    s = cabs1(u)
                    if (s != 0.0) {
                        val x = 0.5.toComplexDouble() * (h[i - 1 + TRANS, i - 1 + TRANS] - t)
                        val sx = cabs1(x)
                        s = max(s, cabs1(x))
                        var y = s.toComplexDouble() * csqrt((x / s) * (x / s) + (u / s) * (u / s))
                        if (sx > 0.0) {
                            if( (x / sx).re * y.re + (x / sx).im * y.im < 0.0) {
                                y = -y
                            }
                        }
                        t -= u * (u / (x + y))
                    }
                }
            }

            // Look for two consecutive small subdiagonal elements
            var h11: ComplexDouble = 0.0.toComplexDouble()
            var h11s: ComplexDouble = 0.0.toComplexDouble()
            var h22: ComplexDouble = 0.0.toComplexDouble()
            var h12: ComplexDouble = 0.0.toComplexDouble()
            var h21: ComplexDouble = 0.0.toComplexDouble()

            var foundm = 1
            var isFound = false

            for (m in (i - 1) downTo (l + 1)) {
                h11 = h[m + TRANS, m + TRANS]
                h22 = h[m + 1 + TRANS, m + 1 + TRANS]
                h11s = h11 - t
                h21 = h[m + 1 + TRANS, m + TRANS].re.toComplexDouble()
                s = cabs1(h11s) + h21.abs()
                h11s /= s
                h21 /= s
                v[1 + TRANS] = h11s
                v[2 + TRANS] = h21
                val h10 = h[m + TRANS, m - 1 + TRANS]

                if (h10.abs() * h21.abs() <= ulp * (cabs1(h11s) * (cabs1(h11) + cabs1(h22)))) {
                    isFound = true
                    foundm = m
                    break
                }
            }
            if (!isFound) {
                h11 = h[l + TRANS, l + TRANS]
                h22 = h[l + 1 + TRANS, l + 1 + TRANS]
                h11s = h11 - t
                h21 = h[l + 1 + TRANS, l + TRANS].re.toComplexDouble()

                s = cabs1(h11s) + h21.abs()
                h11s /= s
                h21 /= s
                v[1 + TRANS] = h11s
                v[2 + TRANS] = h21
                foundm = l
            }

            // single-shift qr step
            for (k in foundm..(i - 1)) {
                if (k > foundm) {
                    v[1 + TRANS] = h[k + TRANS, k - 1 + TRANS]
                    v[2 + TRANS] = h[k + 1 + TRANS, k - 1 + TRANS] //not sure!!!
                }

                // zlarfg( 2, v( 1 ), v( 2 ), 1, t1 )
                val (v1, t1) = computeHouseholderReflectorInline(2, v[1 + TRANS], v[2 + TRANS..3 + TRANS] as D1Array<ComplexDouble>)
                v[1 + TRANS] = v1




                if (k > foundm) {
                    h[k + TRANS, k - 1 + TRANS] = v[1 + TRANS]
                    h[k + 1 + TRANS, k - 1 + TRANS] = 0.0.toComplexDouble()
                }
                // t1 is tau

                val v2 = v[2 + TRANS]
                val t2 = (t1 * v2).re

                //  *           Apply G from the left to transform the rows of the matrix
                // *           in columns K to I2.
                for (j in k..i2) {
                    val sum = t1.conjugate() * h[k + TRANS, j + TRANS] + t2.toComplexDouble() * h[k + 1 + TRANS, j + TRANS]
                    h[k + TRANS, j + TRANS] = h[k + TRANS, j + TRANS] - sum
                    h[k + 1 + TRANS, j + TRANS] = h[k + 1 + TRANS, j + TRANS] - sum * v2
                }

                for (j in i1..min(k + 2, i)) {
                    val sum = t1 * h[j + TRANS, k + TRANS] + t2.toComplexDouble() * h[j + TRANS, k + 1 + TRANS]
                    h[j + TRANS, k + TRANS] -= sum
                    h[j + TRANS, k + 1 + TRANS] -= sum * v2.conjugate()
                }

                for (j in iloz..ihiz) {
                    val sum = t1 * z[j + TRANS, k + TRANS] + t2.toComplexDouble() * z[j + TRANS, k + 1 + TRANS]
                    z[j + TRANS, k + TRANS] -= sum
                    z[j + TRANS, k + 1 + TRANS] -= sum * v2.conjugate()
                }

                if (k == foundm && foundm > l) {
                    var temp = 1.0.toComplexDouble() - t1
                    temp /= temp.abs()
                    h[foundm + 1 + TRANS, foundm + TRANS] = h[foundm + 1 + TRANS, foundm + TRANS] * temp.conjugate()
                    if (foundm + 2 < i) {
                        h[foundm + 2 + TRANS, foundm + 1 + TRANS] *= temp
                    }
                    for (j in foundm..i) {
                        if (j != foundm + 1) {
                            if (i2 > j) {
                                //CALL zscal( i2-j, temp, h( j, j+1 ), ldh )
                                for (ii in 0 until (i2 - j)) {
                                    h[j + TRANS, j + 1 + ii + TRANS] *= temp
                                }
                            }
                            for (ii in 0 until (j - i1)) {
                                h[i1 + ii + TRANS, j + TRANS] *= temp.conjugate()
                            }
                            for (ii in 0 until nz) {
                                z[iloz + ii + TRANS, j + TRANS] *= temp.conjugate()
                            }
                        }
                    }

                }
            }
            // Ensure that H(I,I-1) is real.
            var temp = h[i + TRANS, i - 1 + TRANS]
            if (temp.im != 0.0) {
                val rtemp = temp.abs().toComplexDouble()
                h[i + TRANS, i - 1 + TRANS] = rtemp
                temp /= rtemp
                if (i2 > i) {
                    for (ii in 0 until (i2 - i)) {
                        h[i + TRANS, i + 1 + ii + TRANS] *= temp.conjugate()
                    }
                }
                for (ii in 0 until (i - i1)) {
                    h[i1 + ii + TRANS, i + TRANS] *= temp
                }
                for (ii in 0 until nz) {
                    z[iloz + ii + TRANS, i + TRANS] *= temp
                }
            }
        }

        w[i + TRANS] = h[i + TRANS, i + TRANS]
        kdefl = 0
        i = l - 1
    }


    return Pair(h, z)
}

// return (alpha, tau), mute x
fun computeHouseholderReflectorInline(n: Int, _alpha: ComplexDouble, x: D1Array<ComplexDouble>): Pair<ComplexDouble, ComplexDouble> {
    var alpha = _alpha
    if (n <= 0) {
        return Pair(alpha, ComplexDouble.zero)
    }
    var xnorm = 0.0
    for (i in 0 until n - 1) {
        xnorm += (x[i] * x[i].conjugate()).re
    }
    xnorm = sqrt(max(xnorm, 0.0))
    var alphr = _alpha.re
    var alphi = _alpha.im

    if (xnorm == 0.0 && alphi == 0.0) {
        return Pair(alpha, ComplexDouble.zero)
    }

    var beta = -signum(alphr) * sqrt(alphr * alphr + alphi * alphi + xnorm * xnorm)
    val safmin = 2e-300
    val rsafmn = 1.0 / safmin

    var knt = 0
    while (abs(beta) < safmin) {
        knt++
        for (ii in 0 until n - 1) {
            x[ii] *= rsafmn.toComplexDouble()
        }
        beta *= rsafmn
        alphi *= rsafmn
        alphr *= safmin

        if (abs(beta) < safmin && knt < 20) {
            continue
        }

        xnorm = 0.0
        for (i in 0 until n - 1) {
            xnorm += (x[i] * x[i].conjugate()).re
        }
        xnorm = sqrt(max(xnorm, 0.0))
        alpha = ComplexDouble(alphr, alphi)
        beta = -signum(alphr) * sqrt(alphr * alphr + alphi * alphi + xnorm * xnorm)
    }

    val tau = ComplexDouble((beta - alphr) / beta, -alphi / beta)
    alpha = 1.0.toComplexDouble() / (alpha - beta)

    for (ii in 0 until n - 1) {
        x[ii] *= alpha
    }
    for (j in 1..knt) {
        beta *= safmin
    }
    alpha = beta.toComplexDouble()


    return Pair(alpha, tau)
}

// complex number L1 norm
fun cabs1(a: ComplexDouble): Double {
    return a.re.absoluteValue + a.im.absoluteValue
}


/** sign of number
 *
 * differs from builtin sign:
 * signum(0) = 1
 * sign(0) = 0
 */
fun signum(x: Double): Double {
    return if(x >= 0) 1.0 else -1.0
}
