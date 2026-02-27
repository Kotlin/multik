package org.jetbrains.kotlinx.multik.api.linalg

import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray


/**
 * Linear algebra operations interface, accessible via `mk.linalg`.
 *
 * Provides core matrix operations directly ([pow]) and delegates extended operations
 * (dot, inv, solve, decompositions, eigenvalues, norms) to [LinAlgEx] via [linAlgEx].
 * Convenience extension functions on [LinAlg] are defined in separate files
 * (e.g., [dot], [inv], [eig], [solve]).
 *
 * @see [LinAlgEx] for the full set of extended linear algebra operations.
 */
public interface LinAlg {

    /**
     * The extended linear algebra operations provider.
     */
    public val linAlgEx: LinAlgEx

    /**
     * Raises a square matrix to a non-negative integer power by repeated matrix multiplication.
     *
     * The input [mat] must be square. When [n] is 0, the result is the identity matrix
     * with the same shape.
     *
     * ```
     * val a = mk.ndarray(mk[mk[1, 1], mk[1, 0]])
     * val a3 = mk.linalg.pow(a, 3)
     * // [[3, 2],
     * //  [2, 1]]
     * ```
     *
     * @param mat a square numeric matrix.
     * @param n the exponent (non-negative integer).
     * @return a new [NDArray] of the same shape as [mat].
     * @see [inv] for computing the matrix inverse.
     */
    public fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2>

}
