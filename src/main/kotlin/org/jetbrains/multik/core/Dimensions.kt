package org.jetbrains.multik.core

/**
 * Dimension class.
 */
interface Dimension {
    val d: Int
}


interface DimN : Dimension
interface Dim4 : DimN
interface Dim3 : Dim4
interface Dim2 : Dim3
interface Dim1 : Dim2

@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")
inline fun <D : Dimension> dimensionOf(dim: Int): D = when (dim) {
    1 -> D1
    2 -> D2
    3 -> D3
    4 -> D4
    else -> DN(dim)
} as D

inline fun <reified D : Dimension> dimensionClassOf(dim: Int = -1): D = when (D::class) {
    D1::class -> D1
    D2::class -> D2
    D3::class -> D3
    D4::class -> D4
    else -> DN(dim)
} as D

class DN(override val d: Int) : Dimension, DimN {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }
}

sealed class D4(override val d: Int = 4) : Dimension, Dim4 {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }

    companion object : D4()
}

sealed class D3(override val d: Int = 3) : Dimension, Dim3 {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }

    companion object : D3()
}

sealed class D2(override val d: Int = 2) : Dimension, Dim2 {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }

    companion object : D2()
}

sealed class D1(override val d: Int = 1) : Dimension, Dim1 {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }

    companion object : D1()
}