package org.jetbrains.multik.core

/**
 * Dimension class.
 */
open class DN(val d: Int) {
    companion object : DN(5) {
        @Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")
        inline fun <D : DN> of(dim: Int): D = when (dim) {
            1 -> D1
            2 -> D2
            3 -> D3
            4 -> D4
            else -> DN(dim)
        } as D

        inline fun <reified D : DN> of(): D = when (D::class) {
            D1::class -> D1
            D2::class -> D2
            D3::class -> D3
            D4::class -> D4
            DN::class -> DN
            else -> throw IllegalArgumentException("The dimension ${D::class.simpleName} does't exist.")
        } as D
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = d

    override fun toString(): String {
        return "dimension: $d"
    }
}

sealed class D4(d: Int = 4) : DN(d) {
    companion object : D4()
}

sealed class D3(d: Int = 3) : D4(d) {
    companion object : D3()
}

sealed class D2(d: Int = 2) : D3(d) {
    companion object : D2()
}

sealed class D1(d: Int = 1) : D2(d) {
    companion object : D1()
}