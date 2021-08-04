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
    fun whatewer() {
//        val a = initRandomComplex(5, 5);
//        println(a)
//        val b = upperHessenberg(a).second
//        println(b)
//        println("-------------------------")

        val H = mk.empty<ComplexDouble, D2>(4, 4)
        H[1 - 1, 1 - 1] = ComplexDouble( 1.0, 2.0)
        H[1 - 1, 2 - 1] = ComplexDouble( 3.0, 4.0)
        H[1 - 1, 3 - 1] = ComplexDouble( 5.0, 6.0)
        H[1 - 1, 4 - 1] = ComplexDouble( 7.0, 8.0)
        H[2 - 1, 1 - 1] = ComplexDouble( 9.0,10.0)
        H[2 - 1, 2 - 1] = ComplexDouble(11.0,12.0)
        H[2 - 1, 3 - 1] = ComplexDouble(13.0,14.0)
        H[2 - 1, 4 - 1] = ComplexDouble(15.0,16.0)
        H[3 - 1, 1 - 1] = ComplexDouble( 0.0, 0.0)
        H[3 - 1, 2 - 1] = ComplexDouble(17.0,18.0)
        H[3 - 1, 3 - 1] = ComplexDouble(19.0,20.0)
        H[3 - 1, 4 - 1] = ComplexDouble(21.0,22.0)
        H[4 - 1, 1 - 1] = ComplexDouble( 0.0, 0.0)
        H[4 - 1, 2 - 1] = ComplexDouble( 0.0, 0.0)
        H[4 - 1, 3 - 1] = ComplexDouble(23.0,24.0)
        H[4 - 1, 4 - 1] = ComplexDouble(25.0,26.0)
        println(qrShifted(H).first)

    }


}