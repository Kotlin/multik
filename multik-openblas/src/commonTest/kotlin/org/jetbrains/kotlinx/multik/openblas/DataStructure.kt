package org.jetbrains.kotlinx.multik.openblas

import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import kotlin.random.Random

class DataStructure(private val seed: Int) {

    fun getFloatM(n: Int, m: Int = n): D2Array<Float> {
        val random = Random(seed)
        return mk.d2array(n, m) { random.nextFloat() }
    }

    fun getComplexFloatM(n: Int, m: Int = n): D2Array<ComplexFloat> {
        val random = Random(seed)
        return mk.d2array(n, m) { ComplexFloat(random.nextFloat(), random.nextFloat()) }
    }

    fun getDoubleM(n: Int, m: Int = n): D2Array<Double> {
        val random = Random(seed)
        return mk.d2array(n, m) { random.nextDouble() }
    }

    fun getComplexDoubleM(n: Int, m: Int = n): D2Array<ComplexDouble> {
        val random = Random(seed)
        return mk.d2array(n, m) { ComplexDouble(random.nextDouble(), random.nextDouble()) }
    }

    fun getFloatV(n: Int): D1Array<Float> {
        val random = Random(seed)
        return mk.d1array(n) { random.nextFloat() }
    }

    fun getDoubleV(n: Int): D1Array<Double> {
        val random = Random(seed)
        return mk.d1array(n) { random.nextDouble() }
    }

    fun getFloatMM(
        sizeAD1: Int, sizeAD2: Int = sizeAD1,
        sizeBD1: Int, sizeBD2: Int = sizeBD1
    ): Pair<D2Array<Float>, D2Array<Float>> {
        val random = Random(seed)
        return Pair(
            mk.d2array(sizeAD1, sizeAD2) { random.nextFloat() },
            mk.d2array(sizeBD1, sizeBD2) { random.nextFloat() })
    }

    fun getDoubleMM(
        sizeAD1: Int, sizeAD2: Int = sizeAD1,
        sizeBD1: Int, sizeBD2: Int = sizeBD1
    ): Pair<D2Array<Double>, D2Array<Double>> {
        val random = Random(seed)
        return Pair(
            mk.d2array(sizeAD1, sizeAD2) { random.nextDouble() },
            mk.d2array(sizeBD1, sizeBD2) { random.nextDouble() })
    }

    fun getComplexDoubleMM(
        sizeAD1: Int, sizeAD2: Int = sizeAD1,
        sizeBD1: Int, sizeBD2: Int = sizeBD1
    ): Pair<D2Array<ComplexDouble>, D2Array<ComplexDouble>> {
        val random = Random(seed)
        return Pair(
            mk.d2array(sizeAD1, sizeAD2) { ComplexDouble(random.nextDouble(), random.nextDouble()) },
            mk.d2array(sizeBD1, sizeBD2) { ComplexDouble(random.nextDouble(), random.nextDouble()) })
    }

    fun getFloatVV(sizeA: Int, sizeB: Int = sizeA): Pair<D1Array<Float>, D1Array<Float>> {
        val random = Random(seed)
        return Pair(mk.d1array(sizeA) { random.nextFloat() }, mk.d1array(sizeB) { random.nextFloat() })
    }

    fun getDoubleVV(sizeA: Int, sizeB: Int = sizeA): Pair<D1Array<Double>, D1Array<Double>> {
        val random = Random(seed)
        return Pair(mk.d1array(sizeA) { random.nextDouble() }, mk.d1array(sizeB) { random.nextDouble() })
    }

    fun getDoubleMV(
        sizeMD1: Int,
        sizeMD2: Int = sizeMD1,
        sizeV: Int = sizeMD1
    ): Pair<D2Array<Double>, D1Array<Double>> {
        val random = Random(seed)
        return Pair(mk.d2array(sizeMD1, sizeMD2) { random.nextDouble() }, mk.d1array(sizeV) { random.nextDouble() })
    }

    fun getFloatMV(sizeMD1: Int, sizeMD2: Int = sizeMD1, sizeV: Int = sizeMD1): Pair<D2Array<Float>, D1Array<Float>> {
        val random = Random(seed)
        return Pair(mk.d2array(sizeMD1, sizeMD2) { random.nextFloat() }, mk.d1array(sizeV) { random.nextFloat() })
    }
}