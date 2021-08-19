package org.jetbrains.kotlinx.multik.jni.linalg

internal object JniLinAlg {
    external fun pow(mat: FloatArray, n: Int, result: FloatArray)
    external fun pow(mat: DoubleArray, n: Int, result: DoubleArray)
    external fun norm(mat: FloatArray, p: Int): Double
    external fun norm(mat: DoubleArray, p: Int): Double
    external fun inv(n: Int, mat: FloatArray, strA: Int): Int
    external fun inv(n: Int, mat: DoubleArray, strA: Int): Int
    external fun solve(n: Int, nrhs: Int, a: FloatArray, strA: Int, b: FloatArray, strB: Int): Int
    external fun solve(n: Int, nrhs: Int, a: DoubleArray, strA: Int, b: DoubleArray, strB: Int): Int


    external fun dotMM(transA: Boolean, a: FloatArray, m: Int, n: Int, transB: Boolean, b: FloatArray, k: Int, c: FloatArray)
    external fun dotMM(transA: Boolean, a: DoubleArray, m: Int, n: Int, transB: Boolean, b: DoubleArray, k: Int, c: DoubleArray)
    external fun dotMMC(transA: Boolean, a: FloatArray, m: Int, n: Int, transB: Boolean, b: FloatArray, k: Int, c: FloatArray)
    external fun dotMMC(transA: Boolean, a: DoubleArray, m: Int, n: Int, transB: Boolean, b: DoubleArray, k: Int, c: DoubleArray)
    external fun dotMV(transA: Boolean, a: FloatArray, m: Int, n: Int, b: FloatArray, c: FloatArray)
    external fun dotMV(transA: Boolean, a: DoubleArray, m: Int, n: Int, b: DoubleArray, c: DoubleArray)
    external fun dotMVC(transA: Boolean, a: FloatArray, m: Int, n: Int, b: FloatArray, c: FloatArray)
    external fun dotMVC(transA: Boolean, a: DoubleArray, m: Int, n: Int, b: DoubleArray, c: DoubleArray)
    external fun dotVV(n: Int, a: FloatArray, strideA: Int, b: FloatArray, strideB: Int): Float
    external fun dotVV(n: Int, a: DoubleArray, strideA: Int, b: DoubleArray, strideB: Int): Double
    external fun dotVVC(n: Int, a: FloatArray, strideA: Int, b: FloatArray, strideB: Int): Float
    external fun dotVVC(n: Int, a: DoubleArray, strideA: Int, b: DoubleArray, strideB: Int): Double
}