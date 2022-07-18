/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.openblas.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat

internal expect object JniLinAlg {
    fun pow(mat: FloatArray, n: Int, result: FloatArray)
    fun pow(mat: DoubleArray, n: Int, result: DoubleArray)
    fun norm(norm: Char, m: Int, n: Int, mat: FloatArray, lda: Int): Float
    fun norm(norm: Char, m: Int, n: Int, mat: DoubleArray, lda: Int): Double

    /**
     * @param n number of rows and columns of the matrix [mat]
     * @param mat square matrix
     * @param lda first dimension of the matrix [mat]
     * @return int:
     * = 0 - successful exit
     * < 0 - if number = -i, the i-th argument had an illegal value
     * > 0 if number = i, U(i,i) is exactly zero; the matrix is singular and its inverse could not be computed.
     */
    fun inv(n: Int, mat: FloatArray, lda: Int): Int
    fun inv(n: Int, mat: DoubleArray, lda: Int): Int
    fun invC(n: Int, mat: FloatArray, lda: Int): Int
    fun invC(n: Int, mat: DoubleArray, lda: Int): Int

    fun qr(m: Int, n: Int, qa: FloatArray, lda: Int, r: FloatArray): Int
    fun qr(m: Int, n: Int, qa: DoubleArray, lda: Int, r: DoubleArray): Int
    fun qrC(m: Int, n: Int, qa: FloatArray, lda: Int, r: FloatArray): Int
    fun qrC(m: Int, n: Int, qa: DoubleArray, lda: Int, r: DoubleArray): Int

    fun plu(m: Int, n: Int, a: FloatArray, lda: Int, ipiv: IntArray): Int
    fun plu(m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int
    fun pluC(m: Int, n: Int, a: FloatArray, lda: Int, ipiv: IntArray): Int
    fun pluC(m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int

    fun eig(n: Int, a: FloatArray, w: FloatArray, computeV: Char, vr: FloatArray?): Int
    fun eig(n: Int, a: DoubleArray, w: DoubleArray, computeV: Char, vr: DoubleArray?): Int

    /**
     * @param n
     * @param nrhs
     * @param a
     * @param lda
     * @param b
     * @param ldb
     * @return
     */
    fun solve(n: Int, nrhs: Int, a: FloatArray, lda: Int, b: FloatArray, ldb: Int): Int
    fun solve(n: Int, nrhs: Int, a: DoubleArray, lda: Int, b: DoubleArray, ldb: Int): Int
    fun solveC(n: Int, nrhs: Int, a: FloatArray, lda: Int, b: FloatArray, ldb: Int): Int
    fun solveC(n: Int, nrhs: Int, a: DoubleArray, lda: Int, b: DoubleArray, ldb: Int): Int


    /**
     * @param transA transposed matrix [a]
     * @param offsetA offset of the matrix [a]
     * @param a first matrix
     * @param m number of rows of the matrix [a] and of the matrix [c]
     * @param k number of columns of the matrix [a] and number of rows of the matrix [b]
     * @param lda first dimension of the matrix [a]
     * @param transB transposed matrix [b]
     * @param offsetB offset of the matrix [b]
     * @param b second matrix
     * @param n number of columns of the matrix [b] and of the matrix [c]
     * @param ldb first dimension of the matrix [b]
     * @param c matrix of result
     */
    fun dotMM(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: FloatArray, n: Int, ldb: Int, c: FloatArray)
    fun dotMM(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: DoubleArray, n: Int, ldb: Int, c: DoubleArray)
    fun dotMMC(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: FloatArray, n: Int, ldb: Int, c: FloatArray)
    fun dotMMC(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: DoubleArray, n: Int, ldb: Int, c: DoubleArray)

    /**
     * @param transA transposed matrix [a]
     * @param offsetA offset of the matrix [a]
     * @param a first matrix
     * @param m number of rows of the matrix [a]
     * @param n number of columns of the matrix [a]
     * @param lda first dimension of the matrix [a]
     * @param x vector
     * @param y vector
     */
    fun dotMV(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, n: Int, lda: Int, x: FloatArray, incX: Int, y: FloatArray)
    fun dotMV(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, n: Int, lda: Int, x: DoubleArray, incX: Int, y: DoubleArray)
    fun dotMVC(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, n: Int, lda: Int, x: FloatArray, incX: Int, y: FloatArray)
    fun dotMVC(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, n: Int, lda: Int, x: DoubleArray, incX: Int, y: DoubleArray)

    /**
     * @param n size of vectors
     * @param x first vector
     * @param incX stride of the vector [x]
     * @param y second vector
     * @param incY stride of the vector [y]
     */
    fun dotVV(n: Int, x: FloatArray, incX: Int, y: FloatArray, incY: Int): Float
    fun dotVV(n: Int, x: DoubleArray, incX: Int, y: DoubleArray, incY: Int): Double
    fun dotVVC(n: Int, x: FloatArray, incX: Int, y: FloatArray, incY: Int): ComplexFloat
    fun dotVVC(n: Int, x: DoubleArray, incX: Int, y: DoubleArray, incY: Int): ComplexDouble
}