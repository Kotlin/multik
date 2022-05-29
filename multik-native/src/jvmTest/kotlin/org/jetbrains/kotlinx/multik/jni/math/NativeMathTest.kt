/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jni.math

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.jni.DataStructure
import org.jetbrains.kotlinx.multik.jni.assertFloatingNDArray
import org.jetbrains.kotlinx.multik.jni.assertFloatingNumber
import org.jetbrains.kotlinx.multik.jni.libLoader
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals

class NativeMathTest {

    private lateinit var data: DataStructure
    private lateinit var ndarray: D2Array<Float>

    @BeforeTest
    fun load() {
        libLoader("multik_jni").manualLoad()

        data = DataStructure(42)
        ndarray = data.getFloatM(2)
    }

    @Test
    fun argMaxTest() = assertEquals(3, NativeMath.argMax(ndarray))

    @Test
    fun argMinTest() = assertEquals(0, NativeMath.argMin(ndarray))

    @Test
    fun expTest() {
        val expected = mk.ndarray(
            mk[mk[1.2539709297612778, 1.499692498450485],
                mk[2.47182503414346, 2.647835581718662]]
        )
        assertFloatingNDArray(expected, NativeMathEx.exp(ndarray))
    }

    @Test
    fun logTest() {
        val expected = mk.ndarray(
            mk[mk[-1.4858262962985072, -0.9032262301885379],
                mk[-0.0998681176145879, -0.026608338154056003]]
        )
        assertFloatingNDArray(expected, NativeMathEx.log(ndarray))
    }

    @Test
    fun sinTest() {
        val expected =
            mk.ndarray(mk[mk[0.22438827641771292, 0.39425779276225403], mk[0.7863984442713585, 0.826995590204087]])
        assertFloatingNDArray(expected, NativeMathEx.sin(ndarray))
    }

    @Test
    fun cosTest() {
        val expected =
            mk.ndarray(mk[mk[0.9744998211422555, 0.9189998872939189], mk[0.6177195859349023, 0.5622084077839763]])
        assertFloatingNDArray(expected, NativeMathEx.cos(ndarray))
    }

    @Test
    fun maxTest() = assertEquals(0.97374254f, mk.math.max(ndarray))

    @Test
    fun minTest() = assertEquals(0.22631526f, mk.math.min(ndarray))

    @Test
    fun sumTest() = assertFloatingNumber(2.5102746f, mk.math.sum(ndarray))
}