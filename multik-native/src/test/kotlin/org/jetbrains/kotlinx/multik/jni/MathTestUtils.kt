package org.jetbrains.kotlinx.multik.jni

import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import java.math.BigDecimal
import java.math.RoundingMode

fun <D: Dimension> roundDouble(ndarray: NDArray<Double, D>): NDArray<Double, D> =
    ndarray.map { BigDecimal(it).setScale(2, RoundingMode.HALF_UP).toDouble() }

fun <D: Dimension> roundFloat(ndarray: NDArray<Float, D>): NDArray<Float, D> =
    ndarray.map { BigDecimal(it.toDouble()).setScale(2, RoundingMode.HALF_UP).toFloat() }