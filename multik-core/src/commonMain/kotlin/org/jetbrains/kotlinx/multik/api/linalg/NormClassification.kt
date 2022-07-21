/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.api.linalg

/**
 * Matrix norm types.
 *
 * @param lapackCode value for lapack
 * @property Max max(abs(A(i,j)))
 * @property N1 denotes the  one norm of a matrix (maximum column sum)
 * @property Inf denotes the  infinity norm  of a matrix  (maximum row sum)
 * @property Fro denotes the  Frobenius norm of a matrix (square root of sum of squares)
 */
public enum class Norm(public val lapackCode: Char) {
    Max('M'),
    N1('1'),
    Inf('I'),
    Fro('F')
}