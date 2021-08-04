package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*
import kotlin.math.*


fun signum(x: Double): Double {
    return if(x >= 0) 1.0 else -1.0;
}

// return (alpha, tau), mute x
fun zlarfg(n: Int, _alpha: ComplexDouble, x: D1Array<ComplexDouble>): Pair<ComplexDouble, ComplexDouble> {
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
    val safmin = 2e-292
    val rsafmn = 1.0 / safmin

    var knt = 0
    while (abs(beta) < safmin) {
        knt++
        // CALL zdscal( n-1, rsafmn, x, incx )
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

    var tau = ComplexDouble((beta - alphr) / beta, -alphi / beta)
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

fun cabs1(a: ComplexDouble): Double {
    return a.re.absoluteValue + a.im.absoluteValue
}

fun csqrt(a: ComplexDouble): ComplexDouble {
    val arg = a.angle()
    val absval = a.abs()
    return ComplexDouble(sqrt(absval) * cos(arg / 2), sqrt(absval) * sin(arg / 2))
}

fun qrShifted(a: MultiArray<ComplexDouble, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>> {

    val TRANS = -1

    val w = mk.empty<ComplexDouble, D1>(a.shape[0])
    val z = mk.empty<ComplexDouble, D2>(a.shape[0], a.shape[0])
    val v = mk.empty<ComplexDouble, D1>(2)

    val dat1 = 3.0 / 4.0
    val kexsh = 10
    val n = a.shape[0]
    var h = deepCopyMatrixTmp(a)

    // *     .. Local Arrays ..
    //       COMPLEX*16         V( 2 )
    // *     .. Statement Function definitions ..
    //       cabs1( cdum ) = abs( dble( cdum ) ) + abs( dimag( cdum ) )

    // TODO: corner case
    //       IF( ilo.EQ.ihi ) THEN
    //          w( ilo ) = h( ilo, ilo )
    //          RETURN
    //       END IF

    val jlo = 1
    val jhi = n

    val ilo = 1
    val ihi = n
    val iloz = 1
    val ihiz = n
    val ldz = n

    //       DO 20 I = ILO + 1, IHI
    //         IF( DIMAG( H( I, I-1 ) ).NE.RZERO ) THEN
    //*           ==== The following redundant normalization
    //*           .    avoids problems with both gradual and
    //*           .    sudden underflow in ABS(H(I,I-1)) ====
    //            SC = H( I, I-1 ) / CABS1( H( I, I-1 ) )
    //            SC = DCONJG( SC ) / ABS( SC )
    //            H( I, I-1 ) = ABS( H( I, I-1 ) )
    //            CALL ZSCAL( JHI-I+1, SC, H( I, I ), LDH )
    //            CALL ZSCAL( MIN( JHI, I+1 )-JLO+1, DCONJG( SC ),
    //     $                  H( JLO, I ), 1 )
    //            IF( WANTZ )
    //     $         CALL ZSCAL( IHIZ-ILOZ+1, DCONJG( SC ), Z( ILOZ, I ), 1 )
    //         END IF
    //   20 CONTINUE

    for (i in (ilo + 1)..ihi) {
        if (h[i + TRANS, i - 1 + TRANS].im != 0.0) {
            var sc = h[i + TRANS, i - 1 + TRANS] / cabs1(h[i + TRANS, i - 1 + TRANS])
            sc = sc.conjugate() / sc.abs()
            println(sc)
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

    println("after normalization:")
    println(h)
    println()

    val nh = ihi - ilo + 1
    val nz = ihiz - iloz + 1


    val safemin = 1e-300
    val safemax = 1e300 / safemin
    val ulp = 1e-16 // precision
    val smlnum = safemin * (nh.toDouble() / ulp)


    val i1 = 1
    val i2 = n
    val itmax = 30 * max( 10, nh )
    var kdefl = 0
//


    var i = ihi
    while(true) {
        if (i < ilo) {
            break
        }
        var l = ilo


        for (its in 0..itmax) { // Look for a single small subdiagonal element
            println("its = ${its}")
            println(h)
            println()
            if(its == itmax) {
                break
            }

            var foundk = 0
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

            println("l = $l")

            if (l > ilo) {
                h[l + TRANS, l - 1 + TRANS] = ComplexDouble.zero
                break
            }
            kdefl++

            var s = 0.0
            var t = ComplexDouble.zero
            var u = ComplexDouble.zero

            when {
                kdefl % (2 * kexsh) == 0 -> {
                    println("perform exceptional shift-1")
                    s = dat1*abs(h[i + TRANS, i - 1 + TRANS].re)
                    t = s.toComplexDouble() + h[i + TRANS, i + TRANS]
                }
                kdefl % kexsh == 0 -> {
                    println("perform exceptional shift-2")
                    s = dat1 * abs(h[l + 1 + TRANS, l + TRANS].re)
                    t = s.toComplexDouble() + h[l + TRANS, l + TRANS]
                }
                else -> {
                    println("perform wilkinson shift")
                    t = h[i + TRANS, i + TRANS]
                    u = csqrt(h[i - 1 + TRANS, i + TRANS]) * csqrt(h[i + TRANS, i - 1 + TRANS])
                    println("H[i - 1, i] = ${h[i - 1 + TRANS, i + TRANS]}")
                    println("sqrt(H[i - 1, i]) = ${csqrt(h[i - 1 + TRANS, i + TRANS])}")
                    println("FUUUUUUCKING SQRT U = $u")
                    s = cabs1(u)
                    if (s != 0.0) {
                        var x = 0.5.toComplexDouble() * (h[i - 1 + TRANS, i - 1 + TRANS] - t)
                        var sx = cabs1(x)
                        s = max(s, cabs1(x))
                        var y = s.toComplexDouble() * csqrt((x / s) * (x / s) + (u / s) * (u / s))
                        if (sx > 0.0) {
                            if( (x / sx).re * y.re + (x / sx).im * y.im < 0.0) {
                                y = -y
                            }
                        }
                        t = t - u * (u / (x + y))
                    }
                }
            }
            println("found S = $s")
            println("found T = $t")

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
                var h10 = h[m + TRANS, m - 1 + TRANS]

                //             print *, 'now M = ', M
                //            print *, 'looking for two small'
                //            print *, 'checking condition'
                //            print *, 'that', ABS( H10 )*ABS( H21 )
                //            print *, 'less that', ULP*( CABS1( H11S )*
                //     $          ( CABS1( H11 )+CABS1( H22 ) ) )
                //            ! print *, 'less that', ULP * ( CABS1( H11S ) * ( CABS1(H11)+CABS1(H22)))
                //            IF( ABS( H10 )*ABS( H21 ).LE.ULP*( CABS1( H11S )*
                //     $          ( CABS1( H11 )+CABS1( H22 ) ) ) ) THEN
                //                  print *, 'condition true'
                //            ELSE
                //                  print *, 'its not true'
                //            END IF

                println("now M = $m\nloocking for small\nchecking condition\nthat ${h10.abs() * h21.abs() }")
                println("less that ${ulp*( cabs1( h11s )*( cabs1( h11 )+cabs1( h22 ) ) )}")


                if (h10.abs() * h21.abs() <= ulp * (cabs1(h11s) * (cabs1(h11) + cabs1(h22)))) {
                    println("condition is true")
                    isFound = true
                    foundm = m
                    break
                }
            }
            if (!isFound) {
                println("not found-482")
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
            println("foundm = $foundm")
            println("h11 = $h11")
            println("h11s = $h11s")
            println("h22 = $h22")
            println("h21 = $h21")

            // single-shift qr step
            println("applying single-shift qr step")
            for (k in foundm..(i - 1)) {
                if (k > foundm) {
                    v[1 + TRANS] = h[k + TRANS, k - 1 + TRANS]
                    v[2 + TRANS] = h[k + 1 + TRANS, k - 1 + TRANS] //not sure!!!
                }
//                val vtranspose = mk.empty<ComplexDouble, D2>(2, 1)
//                val extractBeta = mk.empty<ComplexDouble, D2>(2, 1)
//                vtranspose[0, 0] = v[1 + TRANS]
//                vtranspose[1, 0] = v[2 + TRANS]
//                extractBeta[0, 0] = v[1 + TRANS]
//                extractBeta[1, 0] = v[2 + TRANS]

                // zlarfg( 2, v( 1 ), v( 2 ), 1, t1 )
                var (v1, t1) = zlarfg(2, v[1 + TRANS], v[2 + TRANS..3 + TRANS] as D1Array<ComplexDouble>)
                v[1 + TRANS] = v1

/*
                var (tau, vec) = householderTransformComplexDouble(vtranspose) // tau.conj() is t1
                var beta = applyHouseholderComplexDouble(extractBeta, tau, vec)[0, 0] //beta is v(1)
                tau = tau.conjugate()
*/




                println("K =  ${k}")
                println("TAU =  ${t1}")
                println("BETA =  ${v[1 + TRANS]}")
                println("V( 2 ) =  ${v[2 + TRANS]}")

                if (k > foundm) {
                    h[k + TRANS, k - 1 + TRANS] = v[1 + TRANS]
                    h[k + 1 + TRANS, k - 1 + TRANS] = 0.0.toComplexDouble()
                }
                // t1 is tau

                val v2 = v[2 + TRANS]
                val t2 = (t1 * v2).re // not shure in not conjugate

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
                    if (foundm + 2 <= i) {
                        h[foundm + 2 + TRANS, foundm + 1 + TRANS] *= temp
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
            println(h)

        }

        w[i + TRANS] = h[i + TRANS, i + TRANS]
        println("l = $l")
        kdefl = 0
        i = l - 1
    }

    return Pair(w, z)
}

