package org.jetbrains.kotlinx.multik.jni.linalg

import kotlinx.cinterop.toCValues
import kotlinx.cinterop.useContents
import org.jetbrains.kotlinx.multik.cinterop.*
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat

internal actual object JniLinAlg {
    actual fun pow(mat: FloatArray, n: Int, result: FloatArray) {
        TODO()
    }

    actual fun pow(mat: DoubleArray, n: Int, result: DoubleArray) {
        TODO()
    }

    actual fun norm(mat: FloatArray, p: Int): Double {
        TODO()
    }

    actual fun norm(mat: DoubleArray, p: Int): Double {
        TODO()
    }

    /**
     * @param n number of rows and columns of the matrix [mat]
     * @param mat square matrix
     * @param lda first dimension of the matrix [mat]
     * @return int:
     * = 0 - successful exit
     * < 0 - if number = -i, the i-th argument had an illegal value
     * > 0 if number = i, U(i,i) is exactly zero; the matrix is singular and its inverse could not be computed.
     */
    actual fun inv(n: Int, mat: FloatArray, lda: Int): Int = inverse_matrix_float(n, mat.toCValues(), lda)
    actual fun inv(n: Int, mat: DoubleArray, lda: Int): Int = inverse_matrix_double(n, mat.toCValues(), lda)
    actual fun invC(n: Int, mat: FloatArray, lda: Int): Int = inverse_matrix_complex_float(n, mat.toCValues(), lda)
    actual fun invC(n: Int, mat: DoubleArray, lda: Int): Int = inverse_matrix_complex_double(n, mat.toCValues(), lda)

    actual fun qr(m: Int, n: Int, qa: FloatArray, lda: Int, r: FloatArray): Int =
        qr_matrix_float(m, n, qa.toCValues(), lda, r.toCValues())
    actual fun qr(m: Int, n: Int, qa: DoubleArray, lda: Int, r: DoubleArray): Int =
        qr_matrix_double(m, n, qa.toCValues(), lda, r.toCValues())
    actual fun qrC(m: Int, n: Int, qa: FloatArray, lda: Int, r: FloatArray): Int =
        qr_matrix_complex_float(m, n, qa.toCValues(), lda, r.toCValues())
    actual fun qrC(m: Int, n: Int, qa: DoubleArray, lda: Int, r: DoubleArray): Int =
        qr_matrix_complex_double(m, n, qa.toCValues(), lda, r.toCValues())

    actual fun plu(m: Int, n: Int, a: FloatArray, lda: Int, ipiv: IntArray): Int =
        plu_matrix_float(m, n, a.toCValues(), lda, ipiv.toCValues())
    actual fun plu(m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int =
        plu_matrix_double(m, n, a.toCValues(), lda, ipiv.toCValues())
    actual fun pluC(m: Int, n: Int, a: FloatArray, lda: Int, ipiv: IntArray): Int =
        plu_matrix_complex_float(m, n, a.toCValues(), lda, ipiv.toCValues())
    actual fun pluC(m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int =
        plu_matrix_complex_double(m, n, a.toCValues(), lda, ipiv.toCValues())

    actual fun eig(n: Int, a: FloatArray, w: FloatArray, computeV: Char, vr: FloatArray?): Int = TODO()
//        eigen_float(n, a.toCValues(), w.toCValues(), computeV.toByte(), vr?.toCValues())
    actual fun eig(n: Int, a: DoubleArray, w: DoubleArray, computeV: Char, vr: DoubleArray?): Int = TODO()
//        eigen_double(n, a.toCValues(), w.toCValues(), computeV.toByte(), vr?.toCValues())

    /**
     * @param n
     * @param nrhs
     * @param a
     * @param lda
     * @param b
     * @param ldb
     * @return
     */
    actual fun solve(n: Int, nrhs: Int, a: FloatArray, lda: Int, b: FloatArray, ldb: Int): Int =
        solve_linear_system_float(n, nrhs, a.toCValues(), lda, b.toCValues(), ldb)
    actual fun solve(n: Int, nrhs: Int, a: DoubleArray, lda: Int, b: DoubleArray, ldb: Int): Int =
        solve_linear_system_double(n, nrhs, a.toCValues(), lda, b.toCValues(), ldb)
    actual fun solveC(n: Int, nrhs: Int, a: FloatArray, lda: Int, b: FloatArray, ldb: Int): Int =
        solve_linear_system_complex_float(n, nrhs, a.toCValues(), lda, b.toCValues(), ldb)
    actual fun solveC(n: Int, nrhs: Int, a: DoubleArray, lda: Int, b: DoubleArray, ldb: Int): Int =
        solve_linear_system_complex_double(n, nrhs, a.toCValues(), lda, b.toCValues(), ldb)


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
    actual fun dotMM(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: FloatArray, n: Int, ldb: Int, c: FloatArray) =
        matrix_dot_float(transA, offsetA, a.toCValues(), lda, m, n, k, transB, offsetB, b.toCValues(), ldb, c.toCValues())
    actual fun dotMM(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: DoubleArray, n: Int, ldb: Int, c: DoubleArray) =
        matrix_dot_double(transA, offsetA, a.toCValues(), lda, m, n, k, transB, offsetB, b.toCValues(), ldb, c.toCValues())
    actual fun dotMMC(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: FloatArray, n: Int, ldb: Int, c: FloatArray) =
        matrix_dot_complex_float(transA, offsetA, a.toCValues(), lda, m, n, k, transB, offsetB, b.toCValues(), ldb, c.toCValues())
    actual fun dotMMC(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, k: Int, lda: Int, transB: Boolean, offsetB: Int, b: DoubleArray, n: Int, ldb: Int, c: DoubleArray) =
        matrix_dot_complex_double(transA, offsetA, a.toCValues(), lda, m, n, k, transB, offsetB, b.toCValues(), ldb, c.toCValues())

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
    actual fun dotMV(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, n: Int, lda: Int, x: FloatArray, incX: Int, y: FloatArray) =
        matrix_dot_vector_float(transA, offsetA, a.toCValues(), m, n, lda, x.toCValues(), incX, y.toCValues())
    actual fun dotMV(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, n: Int, lda: Int, x: DoubleArray, incX: Int, y: DoubleArray) =
        matrix_dot_vector_double(transA, offsetA, a.toCValues(), m, n, lda, x.toCValues(), incX, y.toCValues())
    actual fun dotMVC(transA: Boolean, offsetA: Int, a: FloatArray, m: Int, n: Int, lda: Int, x: FloatArray, incX: Int, y: FloatArray) =
        matrix_dot_complex_vector_float(transA, offsetA, a.toCValues(), m, n, lda, x.toCValues(), incX, y.toCValues())
    actual fun dotMVC(transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, n: Int, lda: Int, x: DoubleArray, incX: Int, y: DoubleArray) =
        matrix_dot_complex_vector_double(transA, offsetA, a.toCValues(), m, n, lda, x.toCValues(), incX, y.toCValues())

    /**
     * @param n size of vectors
     * @param x first vector
     * @param incX stride of the vector [x]
     * @param y second vector
     * @param incY stride of the vector [y]
     */
    actual fun dotVV(n: Int, x: FloatArray, incX: Int, y: FloatArray, incY: Int): Float =
        vector_dot_float(n, x.toCValues(), incX, y.toCValues(), incY)
    actual fun dotVV(n: Int, x: DoubleArray, incX: Int, y: DoubleArray, incY: Int): Double =
        vector_dot_double(n, x.toCValues(), incX, y.toCValues(), incY)
    actual fun dotVVC(n: Int, x: FloatArray, incX: Int, y: FloatArray, incY: Int): ComplexFloat =
        vector_dot_complex_float(n, x.toCValues(), incX, y.toCValues(), incY).useContents { ComplexFloat(real, imag) }
    actual fun dotVVC(n: Int, x: DoubleArray, incX: Int, y: DoubleArray, incY: Int): ComplexDouble =
        vector_dot_complex_double(n, x.toCValues(), incX, y.toCValues(), incY).useContents { ComplexDouble(real, imag) }
}