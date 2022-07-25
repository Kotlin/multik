/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

/**
 * Matrix norm types.
 *
 * @property lapackCode value for lapack
 */
public enum class Norm(public val lapackCode: Char) {
    /**
     * max(abs(A(i,j)))
     */
    Max('M'),

    /**
     * denotes the  one norm of a matrix (maximum column sum)
     */
    N1('1'),

    /**
     * denotes the  infinity norm  of a matrix  (maximum row sum)
     */
    Inf('I'),

    /**
     * denotes the  Frobenius norm of a matrix (square root of sum of squares)
     */
    Fro('F')
}