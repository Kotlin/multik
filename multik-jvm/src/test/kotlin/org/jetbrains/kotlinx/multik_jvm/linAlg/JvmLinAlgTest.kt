/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik_jvm.linAlg

import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jvm.JvmLinAlg
import kotlin.random.Random
import kotlin.system.measureNanoTime
import kotlin.test.Test
import kotlin.test.assertEquals

class JvmLinAlgTest {

    @Test
    fun `test of norm function with p=1`() {
        val d2arrayDouble1 = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val d2arrayDouble2 = mk.ndarray(mk[mk[-1.0, -2.0], mk[-3.0, -4.0]])

        assertEquals(10.0, mk.linalg.norm(d2arrayDouble1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayDouble2, 1))

        val d2arrayShort1 = mk.ndarray(mk[mk[1.toShort(), 2.toShort()], mk[3.toShort(), 4.toShort()]])
        val d2arrayShort2 = mk.ndarray(mk[mk[(-1).toShort(), (-2).toShort()], mk[(-3).toShort(), (-4).toShort()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayShort1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayShort2, 1))

        val d2arrayInt1 = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val d2arrayInt2 = mk.ndarray(mk[mk[-1, -2], mk[-3, -4]])

        assertEquals(10.0, mk.linalg.norm(d2arrayInt1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayInt2, 1))

        val d2arrayByte1 = mk.ndarray(mk[mk[1.toByte(), 2.toByte()], mk[3.toByte(), 4.toByte()]])
        val d2arrayByte2 = mk.ndarray(mk[mk[(-1).toByte(), (-2).toByte()], mk[(-3).toByte(), (-4).toByte()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayByte1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayByte2, 1))

        val d2arrayFloat1 = mk.ndarray(mk[mk[1.0.toFloat(), 2.0.toFloat()], mk[3.0.toFloat(), 4.0.toFloat()]])
        val d2arrayFloat2 = mk.ndarray(mk[mk[(-1.0).toFloat(), (-2.0).toFloat()], mk[(-3.0).toFloat(), (-4.0).toFloat()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayFloat1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayFloat2, 1))

        val d2arrayLong1 = mk.ndarray(mk[mk[1.toLong(), 2.toLong()], mk[3.toLong(), 4.toLong()]])
        val d2arrayLong2 = mk.ndarray(mk[mk[(-1).toLong(), (-2).toLong()], mk[(-3).toLong(), (-4).toLong()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayLong1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayLong2, 1))

    }

    @Test
    fun `time test`() {
        val matDouble100x100 = mk.d2array<Double>(100, 100) {
            Random(92384).nextDouble()
        }
        val matInt100x100 = mk.d2array<Int>(100, 100) {
            Random(92384).nextInt()
        }
        val matFloat100x100 = mk.d2array(100, 100) {
            Random(92384).nextFloat()
        }

        var res1 = 0.0
        var res2 = 0.0

        var t1 = 0L
        var t2 = 0L

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matDouble100x100, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matDouble100x100, 2)
        }

        println("first method on 100x100 DoubleMat returned $res1, it took $t1 ns")
        println("second method on 100x100 DoubleMat returned $res2, it took $t2 ns")

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matInt100x100, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matInt100x100, 2)
        }
        println("first method on 100x100 IntMat returned $res1, it took $t1 ns")
        println("second method on 100x100 IntMat returned $res2, it took $t2 ns")

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matFloat100x100, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matFloat100x100, 2)
        }
        println("first method on 100x100 FloatMat returned $res1, it took $t1 ns")
        println("second method on 100x100 FloatMat returned $res2, it took $t2 ns")


        println("-----------------------------------------------------------------")

        val matDouble300x300 = mk.d2array<Double>(300, 300) {
            Random(92384).nextDouble()
        }
        val matInt300x300 = mk.d2array<Int>(300, 300) {
            Random(92384).nextInt()
        }
        val matShort300x300 = mk.d2array(300, 300) {
            Random(92384).nextInt().toShort()
        }

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matDouble300x300, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matDouble300x300, 2)
        }
        println("first method on 300x300 DoubleMat returned $res1, it took $t1 ns")
        println("second method on 300x300 DoubleMat returned $res2, it took $t2 ns")

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matInt300x300, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matInt300x300, 2)
        }
        println("first method on 300x300 IntMat returned $res1, it took $t1 ns")
        println("second method on 300x300 IntMat returned $res2, it took $t2 ns")
        println("-----------------------------------------------------------------")

        val matDouble1000x1000 = mk.d2array<Double>(1000, 1000) {
            Random(92384).nextDouble()
        }
        val matInt1000x1000 = mk.d2array<Int>(1000, 1000) {
            Random(923812).nextInt()
        }
        val matFloat1000x1000 = mk.d2array(1000, 1000) {
            Random(923812).nextFloat()
        }

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matDouble1000x1000, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matDouble1000x1000, 2)
        }
        println("first method on 1000x1000 DoubleMat returned $res1, it took $t1 ns")
        println("second method on 1000x1000 DoubleMat returned $res2, it took $t2 ns")

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matInt1000x1000, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matInt1000x1000, 2)
        }
        println("first method on 1000x1000 IntMat returned $res1, it took $t1 ns")
        println("second method on 1000x1000 IntMat returned $res2, it took $t2 ns")

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matFloat1000x1000, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matFloat1000x1000, 2)
        }
        println("first method on 1000x1000 FloatMat returned $res1, it took $t1 ns")
        println("second method on 1000x1000 FloatMat returned $res2, it took $t2 ns")


        println("-----------------------------------------------------------------")


        val matDouble10000x1000 = mk.d2array<Double>(10000, 1000) {
            Random(92384).nextDouble()
        }
        val matInt10000x1000 = mk.d2array<Int>(10000, 1000) {
            Random(923812).nextInt()
        }
        val matFloat10000x1000 = mk.d2array(10000, 1000) {
            Random(923812).nextFloat()
        }

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matDouble10000x1000, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matDouble10000x1000, 2)
        }
        println("first method on 10000x1000 DoubleMat returned $res1, it took $t1 ns")
        println("second method on 10000x1000 DoubleMat returned $res2, it took $t2 ns")

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matInt10000x1000, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matInt10000x1000, 2)
        }
        println("first method on 10000x1000 IntMat returned $res1, it took $t1 ns")
        println("second method on 10000x1000 IntMat returned $res2, it took $t2 ns")

        t1 = measureNanoTime {
            res1 = mk.linalg.norm(matFloat10000x1000, 2)
        }
        t2 = measureNanoTime {
            res2 = JvmLinAlg.testNorm(matFloat10000x1000, 2)
        }
        println("first method on 10000x1000 FloatMat returned ${res1}, it took $t1 ns")
        println("second method on 10000x1000 FloatMat returned $res2, it took $t2 ns")

    }

}