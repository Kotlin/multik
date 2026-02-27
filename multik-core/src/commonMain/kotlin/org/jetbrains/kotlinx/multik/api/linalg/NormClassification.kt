package org.jetbrains.kotlinx.multik.api.linalg

/**
 * Matrix norm types used by [LinAlg.norm][LinAlgEx.norm] to select the norm computation.
 *
 * @property lapackCode the character code passed to LAPACK routines.
 * @see [LinAlgEx.norm]
 * @see [LinAlgEx.normF]
 */
public enum class Norm(public val lapackCode: Char) {
    /**
     * Maximum absolute element value: max(|a_ij|).
     */
    Max('M'),

    /**
     * One-norm (maximum column sum): max over columns of sum(|a_ij|).
     */
    N1('1'),

    /**
     * Infinity norm (maximum row sum): max over rows of sum(|a_ij|).
     */
    Inf('I'),

    /**
     * Frobenius norm (square root of sum of squares): sqrt(sum(|a_ij|^2)). This is the default.
     */
    Fro('F')
}
