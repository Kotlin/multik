package org.jetbrains.kotlinx.multik.jvm.linalg

import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray

internal fun invDouble(a: MultiArray<Double, D2>): D2Array<Double> = solveDouble(a, mk.identity(a.shape[0]))

internal fun invFloat(a: MultiArray<Float, D2>): D2Array<Float> = solveFloat(a, mk.identity(a.shape[0]))

internal fun invComplexDouble(a: MultiArray<ComplexDouble, D2>): D2Array<ComplexDouble> =
    solveComplexDouble(a, mk.identity(a.shape[0]))

internal fun invComplexFloat(a: MultiArray<ComplexFloat, D2>): D2Array<ComplexFloat> =
    solveComplexFloat(a, mk.identity(a.shape[0]))
