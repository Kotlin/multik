package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.Dim2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import kotlin.jvm.JvmName

/**
 * Solves the linear system **Ax = b** for float matrices, preserving single precision.
 *
 * The coefficient matrix [a] must be square and non-singular. The right-hand side [b] can be
 * a vector ([D1]) or a matrix ([D2]) for multiple right-hand sides.
 *
 * @param a the coefficient matrix (n × n).
 * @param b the right-hand side — a vector of length n or a matrix (n × m).
 * @return the solution **x** as [NDArray] of [Float].
 * @see [inv] for computing the matrix inverse directly.
 */
@JvmName("solveF")
public fun <D : Dim2> LinAlg.solve(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> = this.linAlgEx.solveF(a, b)

/**
 * Solves the linear system **Ax = b** for numeric matrices, returning a double-precision result.
 *
 * The coefficient matrix [a] must be square and non-singular. The right-hand side [b] can be
 * a vector ([D1]) or a matrix ([D2]) for multiple right-hand sides.
 *
 * ```
 * val a = mk.ndarray(mk[mk[2.0, 1.0], mk[1.0, 3.0]])
 * val b = mk.ndarray(mk[5.0, 7.0])
 * val x = mk.linalg.solve(a, b) // x ≈ [1.6, 1.8]
 * ```
 *
 * @param a the coefficient matrix (n × n).
 * @param b the right-hand side — a vector of length n or a matrix (n × m).
 * @return the solution **x** as [NDArray] of [Double].
 * @see [inv] for computing the matrix inverse directly.
 */
@JvmName("solveD")
public fun <T : Number, D : Dim2> LinAlg.solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> = this.linAlgEx.solve(a, b)

/**
 * Solves the linear system **Ax = b** for complex matrices.
 *
 * The coefficient matrix [a] must be square and non-singular. The right-hand side [b] can be
 * a vector ([D1]) or a matrix ([D2]) for multiple right-hand sides.
 *
 * @param a the coefficient matrix (n × n).
 * @param b the right-hand side — a vector of length n or a matrix (n × m).
 * @return the solution **x**.
 * @see [inv] for computing the matrix inverse directly.
 */
@JvmName("solveC")
public fun <T : Complex, D : Dim2> LinAlg.solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> = this.linAlgEx.solveC(a, b)
