/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik_jvm.linAlg


import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.jvm.*
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg.dot
import org.jetbrains.kotlinx.multik.jvm.linalg.dotComplexDouble

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDoubleArray
import kotlin.math.min
import org.jetbrains.kotlinx.multik.ndarray.data.*
import java.awt.event.ComponentListener
import java.lang.AssertionError
import java.lang.IllegalArgumentException
import java.nio.channels.ConnectionPendingException
import kotlin.math.abs
import kotlin.math.max

import kotlin.random.Random
import kotlin.test.*

class workAround {

    fun initRandomComplex(n: Int, m: Int): D2Array<ComplexDouble> {
        val rnd = Random(424242)

        val a = mk.empty<ComplexDouble, D2>(n, m)

        for (i in 0 until n) {
            for (j in 0 until m) {
                a[i, j] = ComplexDouble(rnd.nextDouble(), rnd.nextDouble())
            }
        }
        return a
    }


    @Test
    fun `test qr`() {
        val rnd = Random(424242)
        val n = 3
        val m = 3
        val a = mk.empty<ComplexDouble, D2>(n, m)

        for (i in 0 until n) {
            for (j in 0 until m) {
                a[i, j] = ComplexDouble(rnd.nextDouble(), rnd.nextDouble())
            }
        }

//        val (tau, v) = householderTransformComplexDouble(a)

        println(a)
        val (q, r) = qrComplexDouble(a)
//        println("q = \n${q}")
//        println("r = \n${r}")

        println(dotComplexDouble(q, r))


    }


    @Test
    fun `test hessenberg`() {
        val a = initRandomComplex(3, 3)
        val (u, r) = upperHessenberg(a)
        val ut = deepCopyMatrixTmp(u.transpose())
        for (i in 0 until ut.shape[0]) {
            for (j in 0 until ut.shape[1]) {
                ut[i, j] = ut[i, j].conjugate()
            }
        }
        println(a)
        println(dotComplexDouble(dotComplexDouble(u, r), ut))
        println(dotComplexDouble(u, ut))
        println(r)
    }

    @Test
    fun counterexample() {
        fun whatever(x: ComplexDouble) {
            var _x = x
            _x = ComplexDouble(1.0, 1.0)
        }
        val zer = ComplexDouble(0.0, 0.0)
        whatever(zer)
        println(zer)
    }



    @Test
    fun `test eig`() {
        var mat = mk.empty<ComplexDouble, D2>(2, 2)
        mat[0, 0] = ComplexDouble(2.0, 0.0);
        mat[0, 1] = ComplexDouble(1.0, 0.0);
        mat[1, 0] = ComplexDouble(-1.0, 0.0);
        mat[1, 1] = ComplexDouble(0.0, 0.0);
        println(eig(mat).first) // [1, 1]
        mat[0, 0] = ComplexDouble(1.0, 0.0);
        mat[0, 1] = ComplexDouble(1.0, 0.0);
        mat[1, 0] = ComplexDouble(1.0, 0.0);
        mat[1, 1] = ComplexDouble(1.0, 0.0);
        println(eig(mat).first) // [2, 0]
        mat[0, 0] = ComplexDouble(2.0, 0.0);
        mat[0, 1] = ComplexDouble(2.0, 0.0);
        mat[1, 0] = ComplexDouble(-2.0, 0.0);
        mat[1, 1] = ComplexDouble(-2.0, 0.0);
        println(eig(mat).first) // [0, 0]
        mat[0, 0] = ComplexDouble(-2.0, 0.0);
        mat[0, 1] = ComplexDouble(1.0, 0.0);
        mat[1, 0] = ComplexDouble(-1.0, 0.0);
        mat[1, 1] = ComplexDouble(-2.0, 0.0);
        println(eig(mat).first) // [-2 + i, -2 - i]
        mat[0, 0] = ComplexDouble(-1.0, 0.0);
        mat[0, 1] = ComplexDouble(1.0, 0.0);
        mat[1, 0] = ComplexDouble(-2.0, 0.0);
        mat[1, 1] = ComplexDouble(1.0, 0.0);
        println(eig(mat).first) // [-i, i]
        mat[0, 0] = ComplexDouble(0.0, 0.0);
        mat[0, 1] = ComplexDouble(2.0, 0.0);
        mat[1, 0] = ComplexDouble(-2.0, 0.0);
        mat[1, 1] = ComplexDouble(0.0, 0.0);
        println(eig(mat).first) // [2i, -2i]

        mat = mk.empty<ComplexDouble, D2>(3, 3)
        mat[0, 0] = 2.0.toComplexDouble()
        mat[0, 1] = -1.0.toComplexDouble()
        mat[0, 2] = 2.0.toComplexDouble()
        mat[1, 0] = 5.0.toComplexDouble()
        mat[1, 1] = -3.0.toComplexDouble()
        mat[1, 2] = 3.0.toComplexDouble()
        mat[2, 0] = -1.0.toComplexDouble()
        mat[2, 1] = 0.0.toComplexDouble()
        mat[2, 2] = -2.0.toComplexDouble()
        println(eig(mat).first)
    }


    @Test
    fun whatewer() {
        var H = mk.empty<ComplexDouble, D2>(2, 2)
        H[0, 0] = ComplexDouble(0.0, 0.0)
        H[0, 1] = ComplexDouble(-1.0, 0.0)
        H[1, 0] = ComplexDouble(1.0, 0.0)
        H[1, 1] = ComplexDouble(0.0, 0.0)
        var eigs = qrShifted(H).second;
        //println(qrShifted(H).first)
        //println(qrShifted(H).second)
        //println(dotComplexDouble(H, qrShifted(H).second))
        println("eigs = \n$eigs")
        return


//        H[1 - 1, 1 - 1] = ComplexDouble( 1.0, 2.0)
//        H[1 - 1, 2 - 1] = ComplexDouble( 3.0, 4.0)
//        H[1 - 1, 3 - 1] = ComplexDouble( 5.0, 6.0)
//        H[1 - 1, 4 - 1] = ComplexDouble( 120.0, 8.0)
//        H[2 - 1, 1 - 1] = ComplexDouble( 9.0,10.0)
//        H[2 - 1, 2 - 1] = ComplexDouble(11.0,12.0)
//        H[2 - 1, 3 - 1] = ComplexDouble(130.0,14.0)
//        H[2 - 1, 4 - 1] = ComplexDouble(15.0,16.0)
//        H[3 - 1, 1 - 1] = ComplexDouble( 17.0, 18.0)
//        H[3 - 1, 2 - 1] = ComplexDouble(19.0,20.0)
//        H[3 - 1, 3 - 1] = ComplexDouble(210.0,22.0)
//        H[3 - 1, 4 - 1] = ComplexDouble(23.0,24.0)
//        H[4 - 1, 1 - 1] = ComplexDouble( 250.0, 26.0)
//        H[4 - 1, 2 - 1] = ComplexDouble( 27.0, 28.0)
//        H[4 - 1, 3 - 1] = ComplexDouble(29.0,30.0)
//        H[4 - 1, 4 - 1] = ComplexDouble(31.0,32.0)
        println(H)
        var (L, HH) = upperHessenberg(H)
        println(qrShifted(HH))
        return
        var LL = mk.empty<ComplexDouble, D2>(L.shape[0], L.shape[1])

        for (i in 0 until LL.shape[0]) {
            for (j in 0 until LL.shape[1]) {
                LL[i, j] = L[j, i].conjugate()
            }
        }

        println(tempDot(tempDot(L, HH), LL))
        return

        println("------------------------------H---------------------------------")
//        println(H)
        println(upperHessenberg(H).second)
        println("----------------------------------------------------------------")

        println(qrShifted(upperHessenberg(H).second).first)
    //
//        var H = mk.empty<ComplexDouble, D2>(4, 4)
//        H[1 - 1, 1 - 1] = ComplexDouble( 1.0, 2.0)
//        H[1 - 1, 2 - 1] = ComplexDouble( 3.0, 4.0)
//        H[1 - 1, 3 - 1] = ComplexDouble( 5.0, 6.0)
//        H[1 - 1, 4 - 1] = ComplexDouble( 7.0, 8.0)
//        H[2 - 1, 1 - 1] = ComplexDouble( 9.0,10.0)
//        H[2 - 1, 2 - 1] = ComplexDouble(11.0,12.0)
//        H[2 - 1, 3 - 1] = ComplexDouble(13.0,14.0)
//        H[2 - 1, 4 - 1] = ComplexDouble(15.0,16.0)
//        H[3 - 1, 1 - 1] = ComplexDouble( 17.0, 18.0)
//        H[3 - 1, 2 - 1] = ComplexDouble(19.0,20.0)
//        H[3 - 1, 3 - 1] = ComplexDouble(21.0,22.0)
//        H[3 - 1, 4 - 1] = ComplexDouble(23.0,24.0)
//        H[4 - 1, 1 - 1] = ComplexDouble( 25.0, 26.0)
//        H[4 - 1, 2 - 1] = ComplexDouble( 27.0, 28.0)
//        H[4 - 1, 3 - 1] = ComplexDouble(29.0,30.0)
//        H[4 - 1, 4 - 1] = ComplexDouble(31.0,32.0)
//
//        println(H)
//        H = upperHessenberg(H).second
//        for (j in 0 until H.shape[1]) {
//            for (i in j + 2 until H.shape[0]) {
//                H[i, j] = ComplexDouble.zero
//            }
//        }
//        println(H)

//        val a = initRandomComplex(5, 5);
//        println(a)
//        val b = upperHessenberg(a).second
//        println(b)
//        println("-------------------------")

//        val H = mk.empty<ComplexDouble, D2>(4, 4)
//        H[1 - 1, 1 - 1] = ComplexDouble( 1.0, 2.0)
//        H[1 - 1, 2 - 1] = ComplexDouble( 3.0, 4.0)
//        H[1 - 1, 3 - 1] = ComplexDouble( 5.0, 6.0)
//        H[1 - 1, 4 - 1] = ComplexDouble( 7.0, 8.0)
//        H[2 - 1, 1 - 1] = ComplexDouble( 9.0,10.0)
//        H[2 - 1, 2 - 1] = ComplexDouble(11.0,12.0)
//        H[2 - 1, 3 - 1] = ComplexDouble(13.0,14.0)
//        H[2 - 1, 4 - 1] = ComplexDouble(15.0,16.0)
//        H[3 - 1, 1 - 1] = ComplexDouble( 0.0, 0.0)
//        H[3 - 1, 2 - 1] = ComplexDouble(17.0,18.0)
//        H[3 - 1, 3 - 1] = ComplexDouble(19.0,20.0)
//        H[3 - 1, 4 - 1] = ComplexDouble(21.0,22.0)
//        H[4 - 1, 1 - 1] = ComplexDouble( 0.0, 0.0)
//        H[4 - 1, 2 - 1] = ComplexDouble( 0.0, 0.0)
//        H[4 - 1, 3 - 1] = ComplexDouble(23.0,24.0)
//        H[4 - 1, 4 - 1] = ComplexDouble(25.0,26.0)
//        println(qrShifted(H).first)

        //val H = initRandomComplex(100, 100)
//        println(upperHessenberg(H).second)
//        println(qrShifted(H).first)
    }


}


