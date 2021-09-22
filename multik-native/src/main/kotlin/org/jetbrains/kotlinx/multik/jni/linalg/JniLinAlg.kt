package org.jetbrains.kotlinx.multik.jni.linalg

internal object JniLinAlg {
    external fun pow(mat: FloatArray, n: Int, result: FloatArray)
    external fun pow(mat: DoubleArray, n: Int, result: DoubleArray)
    external fun norm(mat: FloatArray, p: Int): Double
    external fun norm(mat: DoubleArray, p: Int): Double

    /**
     * @param n number of rows and columns of the matrix [mat]
     * @param mat square matrix
     * @param lda first dimension of the matrix [mat]
     * @return int:
     * = 0 - successful exit
     * < 0 - if number = -i, the i-th argument had an illegal value
     * > 0 if number = i, U(i,i) is exactly zero; the matrix is singular and its inverse could not be computed.
     */
    external fun inv(n: Int, mat: FloatArray, lda: Int): Int
    external fun inv(n: Int, mat: DoubleArray, lda: Int): Int
    external fun invC(n: Int, mat: FloatArray, lda: Int): Int
    external fun invC(n: Int, mat: DoubleArray, lda: Int): Int

    external fun qr(m: Int, n: Int, qa: FloatArray, lda: Int, r: FloatArray): Int
    external fun qr(m: Int, n: Int, qa: DoubleArray, lda: Int, r: DoubleArray): Int
    external fun qrC(m: Int, n: Int, qa: FloatArray, lda: Int, r: FloatArray): Int
    external fun qrC(m: Int, n: Int, qa: DoubleArray, lda: Int, r: DoubleArray): Int

    external fun plu(m: Int, n: Int, a: FloatArray, lda: Int, ipiv: IntArray): Int
    external fun plu(m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int
    external fun pluC(m: Int, n: Int, a: FloatArray, lda: Int, ipiv: IntArray): Int
    external fun pluC(m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int

    /**
     * @param n
     * @param nrhs
     * @param a
     * @param lda
     * @param b
     * @param ldb
     * @return
     */
    external fun solve(n: Int, nrhs: Int, a: FloatArray, lda: Int, b: FloatArray, ldb: Int): Int
    external fun solve(n: Int, nrhs: Int, a: DoubleArray, lda: Int, b: DoubleArray, ldb: Int): Int
    external fun solveC(n: Int, nrhs: Int, a: FloatArray, lda: Int, b: FloatArray, ldb: Int): Int
    external fun solveC(n: Int, nrhs: Int, a: DoubleArray, lda: Int, b: DoubleArray, ldb: Int): Int


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
    external fun dotMM(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: FloatArray, n: Int, ldb: Int, c: FloatArray)
    external fun dotMM(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: DoubleArray, n: Int, ldb: Int, c: DoubleArray)
    external fun dotMMC(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: FloatArray, n: Int, ldb: Int, c: FloatArray)
    external fun dotMMC(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: DoubleArray, n: Int, ldb: Int, c: DoubleArray)

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
    external fun dotMV(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, n: Int, lda: Int, x: FloatArray, incX: Int, y: FloatArray)
    external fun dotMV(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, n: Int, lda: Int, x: DoubleArray, incX: Int, y: DoubleArray)
    external fun dotMVC(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, n: Int, lda: Int, x: FloatArray, incX: Int, y: FloatArray)
    external fun dotMVC(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, n: Int, lda: Int, x: DoubleArray, incX: Int, y: DoubleArray)

    /**
     * @param n size of vectors
     * @param x first vector
     * @param incX stride of the vector [x]
     * @param y second vector
     * @param incY stride of the vector [y]
     */
    external fun dotVV(n: Int, x: FloatArray, incX: Int, y: FloatArray, incY: Int): Float
    external fun dotVV(n: Int, x: DoubleArray, incX: Int, y: DoubleArray, incY: Int): Double
    external fun dotVVC(n: Int, x: FloatArray, incX: Int, y: FloatArray, incY: Int): Float
    external fun dotVVC(n: Int, x: DoubleArray, incX: Int, y: DoubleArray, incY: Int): Double
}