package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.api.ExperimentalMultikApi
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*

/**
 * Extended linear algebra operations interface providing type-specific overloads.
 *
 * Each operation has up to three variants by element type: numeric (`Number`), float-specific (`Float`),
 * and complex (`Complex`). Float variants (suffixed `F`) preserve single precision; numeric variants
 * promote to `Double`. Engine implementations (`KEEngine`, `NativeEngine`) implement this interface.
 *
 * Users typically call the convenience extensions on [LinAlg] (e.g., `mk.linalg.dot(a, b)`)
 * rather than accessing this interface directly.
 *
 * @see [LinAlg] for the main entry point.
 */
public interface LinAlgEx {

    // ---- Inverse ----

    /**
     * Computes the inverse of a numeric matrix, returning a double-precision result.
     *
     * @param mat a square, non-singular matrix.
     * @return the inverse matrix as [NDArray] of [Double].
     */
    public fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2>

    /**
     * Computes the inverse of a float matrix, preserving single precision.
     *
     * @param mat a square, non-singular float matrix.
     * @return the inverse matrix as [NDArray] of [Float].
     */
    public fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2>

    /**
     * Computes the inverse of a complex matrix.
     *
     * @param mat a square, non-singular complex matrix.
     * @return the inverse matrix.
     */
    public fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2>

    // ---- Solve ----

    /**
     * Solves the linear system **Ax = b** for numeric matrices, returning a double-precision result.
     *
     * @param a the coefficient matrix (square, non-singular).
     * @param b the right-hand side — a vector ([D1]) or matrix ([D2]).
     * @return the solution **x** as [NDArray] of [Double].
     */
    public fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D>

    /**
     * Solves the linear system **Ax = b** for float matrices, preserving single precision.
     *
     * @param a the coefficient matrix (square, non-singular).
     * @param b the right-hand side — a vector ([D1]) or matrix ([D2]).
     * @return the solution **x** as [NDArray] of [Float].
     */
    public fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D>

    /**
     * Solves the linear system **Ax = b** for complex matrices.
     *
     * @param a the coefficient matrix (square, non-singular).
     * @param b the right-hand side — a vector ([D1]) or matrix ([D2]).
     * @return the solution **x**.
     */
    public fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D>

    // ---- Norm ----

    /**
     * Computes a matrix norm for a float matrix.
     *
     * @param mat the input matrix.
     * @param norm the [Norm] type to compute (default: [Norm.Fro]).
     * @return the scalar norm value.
     */
    public fun normF(mat: MultiArray<Float, D2>, norm: Norm = Norm.Fro): Float

    /**
     * Computes a matrix norm for a double matrix.
     *
     * @param mat the input matrix.
     * @param norm the [Norm] type to compute (default: [Norm.Fro]).
     * @return the scalar norm value.
     */
    public fun norm(mat: MultiArray<Double, D2>, norm: Norm = Norm.Fro): Double

    // ---- QR decomposition ----

    /**
     * Computes the QR decomposition of a numeric matrix into orthogonal **Q** and upper-triangular **R**.
     *
     * @param mat the input matrix.
     * @return a [Pair] of (Q, R) as double-precision [D2Array]s.
     */
    public fun <T : Number> qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>>

    /**
     * Computes the QR decomposition of a float matrix, preserving single precision.
     *
     * @param mat the input matrix.
     * @return a [Pair] of (Q, R) as float [D2Array]s.
     */
    public fun qrF(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>>

    /**
     * Computes the QR decomposition of a complex matrix.
     *
     * @param mat the input matrix.
     * @return a [Pair] of (Q, R) as complex [D2Array]s.
     */
    public fun <T : Complex> qrC(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>>

    // ---- PLU decomposition ----

    /**
     * Computes the PLU decomposition of a numeric matrix such that **P * A = L * U**.
     *
     * @param mat the input matrix.
     * @return a [Triple] of (P, L, U) as double-precision [D2Array]s.
     */
    public fun <T : Number> plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>>

    /**
     * Computes the PLU decomposition of a float matrix, preserving single precision.
     *
     * @param mat the input matrix.
     * @return a [Triple] of (P, L, U) as float [D2Array]s.
     */
    public fun pluF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>>

    /**
     * Computes the PLU decomposition of a complex matrix.
     *
     * @param mat the input matrix.
     * @return a [Triple] of (P, L, U) as complex [D2Array]s.
     */
    public fun <T : Complex> pluC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>>

    // ---- SVD decomposition ----

    /**
     * Computes the SVD decomposition of a float matrix such that **A = U * S * V^T**.
     *
     * @param mat the input matrix.
     * @return a [Triple] of (U, S, V^T) where S is a [D1Array] of singular values.
     */
    @ExperimentalMultikApi
    public fun svdF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D1Array<Float>, D2Array<Float>>

    /**
     * Computes the SVD decomposition of a numeric matrix such that **A = U * S * V^T**.
     *
     * @param mat the input matrix.
     * @return a [Triple] of (U, S, V^T) as double-precision arrays where S is a [D1Array] of singular values.
     */
    @ExperimentalMultikApi
    public fun <T : Number> svd(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D1Array<Double>, D2Array<Double>>

    /**
     * Computes the SVD decomposition of a complex matrix such that **A = U * S * V^H**.
     *
     * @param mat the input matrix.
     * @return a [Triple] of (U, S, V^H) where S is a [D1Array] of singular values.
     */
    @ExperimentalMultikApi
    public fun <T : Complex> svdC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D1Array<T>, D2Array<T>>

    // ---- Eigenvalues and eigenvectors ----

    /**
     * Computes the eigenvalues and eigenvectors of a square numeric matrix.
     *
     * @param mat a square matrix.
     * @return a [Pair] of (eigenvalues, eigenvectors) as [ComplexDouble] arrays.
     */
    public fun <T : Number> eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>>

    /**
     * Computes the eigenvalues and eigenvectors of a square float matrix.
     *
     * @param mat a square float matrix.
     * @return a [Pair] of (eigenvalues, eigenvectors) as [ComplexFloat] arrays.
     */
    public fun eigF(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>>

    /**
     * Computes the eigenvalues and eigenvectors of a square complex matrix.
     *
     * @param mat a square complex matrix.
     * @return a [Pair] of (eigenvalues, eigenvectors).
     */
    public fun <T : Complex> eigC(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>>

    /**
     * Computes only the eigenvalues of a square numeric matrix (without eigenvectors).
     *
     * @param mat a square matrix.
     * @return a [D1Array] of [ComplexDouble] eigenvalues.
     */
    public fun <T : Number> eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble>

    /**
     * Computes only the eigenvalues of a square float matrix (without eigenvectors).
     *
     * @param mat a square float matrix.
     * @return a [D1Array] of [ComplexFloat] eigenvalues.
     */
    public fun eigValsF(mat: MultiArray<Float, D2>): D1Array<ComplexFloat>

    /**
     * Computes only the eigenvalues of a square complex matrix (without eigenvectors).
     *
     * @param mat a square complex matrix.
     * @return a [D1Array] of complex eigenvalues.
     */
    public fun <T : Complex> eigValsC(mat: MultiArray<T, D2>): D1Array<T>

    // ---- Dot product ----

    /**
     * Computes the matrix product of two numeric matrices.
     *
     * @param a the left-hand matrix (m × n).
     * @param b the right-hand matrix (n × p).
     * @return a new [NDArray] of shape (m × p).
     */
    public fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>

    /**
     * Computes the matrix product of two complex matrices.
     *
     * @param a the left-hand matrix (m × n).
     * @param b the right-hand matrix (n × p).
     * @return a new [NDArray] of shape (m × p).
     */
    public fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>

    /**
     * Computes the product of a numeric matrix and a numeric vector.
     *
     * @param a the matrix (m × n).
     * @param b the vector of length n.
     * @return a new [NDArray] vector of length m.
     */
    public fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>

    /**
     * Computes the product of a complex matrix and a complex vector.
     *
     * @param a the matrix (m × n).
     * @param b the vector of length n.
     * @return a new [NDArray] vector of length m.
     */
    public fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>

    /**
     * Computes the inner (scalar) product of two numeric vectors.
     *
     * @param a the first vector.
     * @param b the second vector, same length as [a].
     * @return the scalar product.
     */
    public fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T

    /**
     * Computes the inner (scalar) product of two complex vectors.
     *
     * @param a the first vector.
     * @param b the second vector, same length as [a].
     * @return the scalar product.
     */
    public fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
}
