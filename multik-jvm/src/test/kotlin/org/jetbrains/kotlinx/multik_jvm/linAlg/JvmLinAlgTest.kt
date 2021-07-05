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



}