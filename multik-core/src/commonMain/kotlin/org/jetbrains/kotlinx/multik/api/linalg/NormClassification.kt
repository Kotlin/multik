package org.jetbrains.kotlinx.multik.api.linalg

public enum class Norm(public val lapackCode: Char) {
    Max('M'),
    N1('1'),
    Inf('I'),
    Fro('F')
}