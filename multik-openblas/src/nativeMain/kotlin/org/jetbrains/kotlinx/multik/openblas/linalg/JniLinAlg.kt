package org.jetbrains.kotlinx.multik.openblas.linalg

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.useContents
import kotlinx.cinterop.usePinned
import org.jetbrains.kotlinx.multik.cinterop.inverse_matrix_complex_double
import org.jetbrains.kotlinx.multik.cinterop.inverse_matrix_complex_float
import org.jetbrains.kotlinx.multik.cinterop.inverse_matrix_double
import org.jetbrains.kotlinx.multik.cinterop.inverse_matrix_float
import org.jetbrains.kotlinx.multik.cinterop.matrix_dot_complex_double
import org.jetbrains.kotlinx.multik.cinterop.matrix_dot_complex_float
import org.jetbrains.kotlinx.multik.cinterop.matrix_dot_complex_vector_double
import org.jetbrains.kotlinx.multik.cinterop.matrix_dot_complex_vector_float
import org.jetbrains.kotlinx.multik.cinterop.matrix_dot_double
import org.jetbrains.kotlinx.multik.cinterop.matrix_dot_float
import org.jetbrains.kotlinx.multik.cinterop.matrix_dot_vector_double
import org.jetbrains.kotlinx.multik.cinterop.matrix_dot_vector_float
import org.jetbrains.kotlinx.multik.cinterop.norm_matrix_double
import org.jetbrains.kotlinx.multik.cinterop.norm_matrix_float
import org.jetbrains.kotlinx.multik.cinterop.plu_matrix_complex_double
import org.jetbrains.kotlinx.multik.cinterop.plu_matrix_complex_float
import org.jetbrains.kotlinx.multik.cinterop.plu_matrix_double
import org.jetbrains.kotlinx.multik.cinterop.plu_matrix_float
import org.jetbrains.kotlinx.multik.cinterop.qr_matrix_complex_double
import org.jetbrains.kotlinx.multik.cinterop.qr_matrix_complex_float
import org.jetbrains.kotlinx.multik.cinterop.qr_matrix_double
import org.jetbrains.kotlinx.multik.cinterop.qr_matrix_float
import org.jetbrains.kotlinx.multik.cinterop.solve_linear_system_complex_double
import org.jetbrains.kotlinx.multik.cinterop.solve_linear_system_complex_float
import org.jetbrains.kotlinx.multik.cinterop.solve_linear_system_double
import org.jetbrains.kotlinx.multik.cinterop.solve_linear_system_float
import org.jetbrains.kotlinx.multik.cinterop.vector_dot_complex_double
import org.jetbrains.kotlinx.multik.cinterop.vector_dot_complex_float
import org.jetbrains.kotlinx.multik.cinterop.vector_dot_double
import org.jetbrains.kotlinx.multik.cinterop.vector_dot_float
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat

@OptIn(ExperimentalForeignApi::class)
internal actual object JniLinAlg {
    actual fun pow(mat: FloatArray, n: Int, result: FloatArray) {
        TODO()
    }

    actual fun pow(mat: DoubleArray, n: Int, result: DoubleArray) {
        TODO()
    }

    actual fun norm(norm: Char, m: Int, n: Int, mat: FloatArray, lda: Int): Float = mat.usePinned {
        norm_matrix_float(norm.code.toByte(), m, n, it.addressOf(0), lda)
    }

    actual fun norm(norm: Char, m: Int, n: Int, mat: DoubleArray, lda: Int): Double = mat.usePinned {
        norm_matrix_double(norm.code.toByte(), m, n, it.addressOf(0), lda)
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
    actual fun inv(n: Int, mat: FloatArray, lda: Int): Int = mat.usePinned {
        inverse_matrix_float(n, it.addressOf(0), lda)
    }

    actual fun inv(n: Int, mat: DoubleArray, lda: Int): Int = mat.usePinned {
        inverse_matrix_double(n, it.addressOf(0), lda)
    }

    actual fun invC(n: Int, mat: FloatArray, lda: Int): Int = mat.usePinned {
        inverse_matrix_complex_float(n, it.addressOf(0), lda)
    }

    actual fun invC(n: Int, mat: DoubleArray, lda: Int): Int = mat.usePinned {
        inverse_matrix_complex_double(n, it.addressOf(0), lda)
    }

    actual fun qr(m: Int, n: Int, qa: FloatArray, lda: Int, r: FloatArray): Int =
        qa.usePinned { qaPinned ->
            r.usePinned { rPinned ->
                qr_matrix_float(m, n, qaPinned.addressOf(0), lda, rPinned.addressOf(0))
            }
        }

    actual fun qr(m: Int, n: Int, qa: DoubleArray, lda: Int, r: DoubleArray): Int =
        qa.usePinned { qaPinned ->
            r.usePinned { rPinned ->
                qr_matrix_double(m, n, qaPinned.addressOf(0), lda, rPinned.addressOf(0))
            }
        }

    actual fun qrC(m: Int, n: Int, qa: FloatArray, lda: Int, r: FloatArray): Int =
        qa.usePinned { qaPinned ->
            r.usePinned { rPinned ->
                qr_matrix_complex_float(m, n, qaPinned.addressOf(0), lda, rPinned.addressOf(0))
            }
        }

    actual fun qrC(m: Int, n: Int, qa: DoubleArray, lda: Int, r: DoubleArray): Int =
        qa.usePinned { qaPinned ->
            r.usePinned { rPinned ->
                qr_matrix_complex_double(m, n, qaPinned.addressOf(0), lda, rPinned.addressOf(0))
            }
        }

    actual fun plu(m: Int, n: Int, a: FloatArray, lda: Int, ipiv: IntArray): Int =
        a.usePinned { aPinned ->
            ipiv.usePinned { ipivPinned ->
                plu_matrix_float(m, n, aPinned.addressOf(0), lda, ipivPinned.addressOf(0))
            }
        }

    actual fun plu(m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int =
        a.usePinned { aPinned ->
            ipiv.usePinned { ipivPinned ->
                plu_matrix_double(m, n, aPinned.addressOf(0), lda, ipivPinned.addressOf(0))
            }
        }

    actual fun pluC(m: Int, n: Int, a: FloatArray, lda: Int, ipiv: IntArray): Int =
        a.usePinned { aPinned ->
            ipiv.usePinned { ipivPinned ->
                plu_matrix_complex_float(m, n, aPinned.addressOf(0), lda, ipivPinned.addressOf(0))
            }
        }

    actual fun pluC(m: Int, n: Int, a: DoubleArray, lda: Int, ipiv: IntArray): Int =
        a.usePinned { aPinned ->
            ipiv.usePinned { ipivPinned ->
                plu_matrix_complex_double(m, n, aPinned.addressOf(0), lda, ipivPinned.addressOf(0))
            }
        }

    actual fun svd(
        m: Int,
        n: Int,
        a: FloatArray,
        lda: Int,
        s: FloatArray,
        u: FloatArray,
        ldu: Int,
        vt: FloatArray,
        ldvt: Int
    ): Int =
        TODO("requires quadmath")

    actual fun svd(
        m: Int,
        n: Int,
        a: DoubleArray,
        lda: Int,
        s: DoubleArray,
        u: DoubleArray,
        ldu: Int,
        vt: DoubleArray,
        ldvt: Int
    ): Int =
        TODO("requires quadmath")

    actual fun svdC(
        m: Int,
        n: Int,
        a: FloatArray,
        lda: Int,
        s: FloatArray,
        u: FloatArray,
        ldu: Int,
        vt: FloatArray,
        ldvt: Int
    ): Int =
        TODO("requires quadmath")

    actual fun svdC(
        m: Int,
        n: Int,
        a: DoubleArray,
        lda: Int,
        s: DoubleArray,
        u: DoubleArray,
        ldu: Int,
        vt: DoubleArray,
        ldvt: Int
    ): Int =
        TODO("requires quadmath")

    actual fun eig(n: Int, a: FloatArray, w: FloatArray, computeV: Char, vr: FloatArray?): Int = TODO()
    actual fun eig(n: Int, a: DoubleArray, w: DoubleArray, computeV: Char, vr: DoubleArray?): Int = TODO()

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
        a.usePinned { aPinned ->
            b.usePinned { bPinned ->
                solve_linear_system_float(n, nrhs, aPinned.addressOf(0), lda, bPinned.addressOf(0), ldb)
            }
        }

    actual fun solve(n: Int, nrhs: Int, a: DoubleArray, lda: Int, b: DoubleArray, ldb: Int): Int =
        a.usePinned { aPinned ->
            b.usePinned { bPinned ->
                solve_linear_system_double(n, nrhs, aPinned.addressOf(0), lda, bPinned.addressOf(0), ldb)
            }
        }

    actual fun solveC(n: Int, nrhs: Int, a: FloatArray, lda: Int, b: FloatArray, ldb: Int): Int =
        a.usePinned { aPinned ->
            b.usePinned { bPinned ->
                solve_linear_system_complex_float(n, nrhs, aPinned.addressOf(0), lda, bPinned.addressOf(0), ldb)
            }
        }

    actual fun solveC(n: Int, nrhs: Int, a: DoubleArray, lda: Int, b: DoubleArray, ldb: Int): Int =
        a.usePinned { aPinned ->
            b.usePinned { bPinned ->
                solve_linear_system_complex_double(n, nrhs, aPinned.addressOf(0), lda, bPinned.addressOf(0), ldb)
            }
        }


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
    actual fun dotMM(
        transA: Boolean, offsetA: Int, a: FloatArray, m: Int, k: Int, lda: Int,
        transB: Boolean, offsetB: Int, b: FloatArray, n: Int, ldb: Int, c: FloatArray
    ) = a.usePinned { aPinned ->
        b.usePinned { bPinned ->
            c.usePinned { cPinned ->
                matrix_dot_float(
                    transA, offsetA, aPinned.addressOf(0), lda, m, n, k,
                    transB, offsetB, bPinned.addressOf(0), ldb, cPinned.addressOf(0)
                )
            }
        }
    }

    actual fun dotMM(
        transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, k: Int, lda: Int,
        transB: Boolean, offsetB: Int, b: DoubleArray, n: Int, ldb: Int, c: DoubleArray
    ) = a.usePinned { aPinned ->
        b.usePinned { bPinned ->
            c.usePinned { cPinned ->
                matrix_dot_double(
                    transA, offsetA, aPinned.addressOf(0), lda, m, n, k,
                    transB, offsetB, bPinned.addressOf(0), ldb, cPinned.addressOf(0)
                )
            }
        }
    }

    actual fun dotMMC(
        transA: Boolean, offsetA: Int, a: FloatArray, m: Int, k: Int, lda: Int,
        transB: Boolean, offsetB: Int, b: FloatArray, n: Int, ldb: Int, c: FloatArray
    ) = a.usePinned { aPinned ->
        b.usePinned { bPinned ->
            c.usePinned { cPinned ->
                matrix_dot_complex_float(
                    transA, offsetA, aPinned.addressOf(0), lda, m, n, k,
                    transB, offsetB, bPinned.addressOf(0), ldb, cPinned.addressOf(0)
                )
            }
        }
    }

    actual fun dotMMC(
        transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, k: Int, lda: Int,
        transB: Boolean, offsetB: Int, b: DoubleArray, n: Int, ldb: Int, c: DoubleArray
    ) = a.usePinned { aPinned ->
        b.usePinned { bPinned ->
            c.usePinned { cPinned ->
                matrix_dot_complex_double(
                    transA, offsetA, aPinned.addressOf(0), lda, m, n, k,
                    transB, offsetB, bPinned.addressOf(0), ldb, cPinned.addressOf(0)
                )
            }
        }
    }

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
    actual fun dotMV(
        transA: Boolean, offsetA: Int, a: FloatArray, m: Int, n: Int,
        lda: Int, offsetX: Int, x: FloatArray, incX: Int, y: FloatArray
    ) = a.usePinned { aPinned ->
        x.usePinned { xPinned ->
            y.usePinned { yPinned ->
                matrix_dot_vector_float(
                    transA, offsetA, aPinned.addressOf(0), lda, m, n,
                    offsetX, xPinned.addressOf(0), incX, yPinned.addressOf(0)
                )
            }
        }
    }

    actual fun dotMV(
        transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, n: Int,
        lda: Int, offsetX: Int, x: DoubleArray, incX: Int, y: DoubleArray
    ) = a.usePinned { aPinned ->
        x.usePinned { xPinned ->
            y.usePinned { yPinned ->
                matrix_dot_vector_double(
                    transA, offsetA, aPinned.addressOf(0), lda, m, n,
                    offsetX, xPinned.addressOf(0), incX, yPinned.addressOf(0)
                )
            }
        }
    }

    actual fun dotMVC(
        transA: Boolean, offsetA: Int, a: FloatArray, m: Int, n: Int,
        lda: Int, offsetX: Int, x: FloatArray, incX: Int, y: FloatArray
    ) = a.usePinned { aPinned ->
        x.usePinned { xPinned ->
            y.usePinned { yPinned ->
                matrix_dot_complex_vector_float(
                    transA, offsetA, aPinned.addressOf(0), lda, m, n,
                    offsetX, xPinned.addressOf(0), incX, yPinned.addressOf(0)
                )
            }
        }
    }

    actual fun dotMVC(
        transA: Boolean, offsetA: Int, a: DoubleArray, m: Int, n: Int,
        lda: Int, offsetX: Int, x: DoubleArray, incX: Int, y: DoubleArray
    ) = a.usePinned { aPinned ->
        x.usePinned { xPinned ->
            y.usePinned { yPinned ->
                matrix_dot_complex_vector_double(
                    transA, offsetA, aPinned.addressOf(0), lda, m, n,
                    offsetX, xPinned.addressOf(0), incX, yPinned.addressOf(0)
                )
            }
        }
    }

    /**
     * @param n size of vectors
     * @param x first vector
     * @param incX stride of the vector [x]
     * @param y second vector
     * @param incY stride of the vector [y]
     */
    actual fun dotVV(n: Int, offsetX: Int, x: FloatArray, incX: Int, offsetY: Int, y: FloatArray, incY: Int): Float =
        x.usePinned { xPinned ->
            y.usePinned { yPinned ->
                vector_dot_float(n, offsetX, xPinned.addressOf(0), incX, offsetY, yPinned.addressOf(0), incY)
            }
        }

    actual fun dotVV(n: Int, offsetX: Int, x: DoubleArray, incX: Int, offsetY: Int, y: DoubleArray, incY: Int): Double =
        x.usePinned { xPinned ->
            y.usePinned { yPinned ->
                vector_dot_double(n, offsetX, xPinned.addressOf(0), incX, offsetY, yPinned.addressOf(0), incY)
            }
        }

    actual fun dotVVC(
        n: Int, offsetX: Int, x: FloatArray, incX: Int, offsetY: Int, y: FloatArray, incY: Int
    ): ComplexFloat = x.usePinned { xPinned ->
        y.usePinned { yPinned ->
            vector_dot_complex_float(
                n, offsetX, xPinned.addressOf(0), incX,
                offsetY, yPinned.addressOf(0), incY
            ).useContents { ComplexFloat(real, imag) }
        }
    }

    actual fun dotVVC(
        n: Int, offsetX: Int, x: DoubleArray, incX: Int, offsetY: Int, y: DoubleArray, incY: Int
    ): ComplexDouble = x.usePinned { xPinned ->
        y.usePinned { yPinned ->
            vector_dot_complex_double(
                n, offsetX, xPinned.addressOf(0), incX,
                offsetY, yPinned.addressOf(0), incY
            ).useContents { ComplexDouble(real, imag) }
        }
    }
}
