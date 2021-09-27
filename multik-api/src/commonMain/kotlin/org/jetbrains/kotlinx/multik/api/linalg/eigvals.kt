package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import kotlin.jvm.JvmName

//@JvmName("eigF")
//public fun LinAlg.eig(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>> =
//    this.linAlgEx.eigF(mat)

//@JvmName("eig")
//public fun <T : Number> LinAlg.eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>> =
//    this.linAlgEx.eig(mat)

//@JvmName("eigC")
//public fun <T : Complex> LinAlg.eig(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>> =
//    this.linAlgEx.eigC(mat)

@JvmName("eigValsF")
public fun LinAlg.eigVals(mat: MultiArray<Float, D2>): D1Array<ComplexFloat> = this.linAlgEx.eigValsF(mat)

@JvmName("eigVals")
public fun <T : Number> LinAlg.eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble> = this.linAlgEx.eigVals(mat)

@JvmName("eigValsC")
public fun <T : Complex> LinAlg.eigVals(mat: MultiArray<T, D2>): D1Array<T> = this.linAlgEx.eigValsC(mat)