package org.jetbrains.kotlinx.multik.ndarray.data

/**
 * Marker interface for array dimensionality.
 *
 * The dimension type hierarchy enables covariant bounds:
 * [Dim1] : [Dim2] : [Dim3] : [Dim4] : [DimN] : [Dimension], so a function accepting
 * `D : Dim2` also accepts [D1] arrays.
 *
 * @property d the number of dimensions (rank).
 */
public interface Dimension {
    public val d: Int
}

/** Marker for N-dimensional arrays (rank 5+, or unknown rank). */
public interface DimN : Dimension

/** Marker for arrays with 4 or more dimensions. */
public interface Dim4 : DimN

/** Marker for arrays with 3 or more dimensions. */
public interface Dim3 : Dim4

/** Marker for arrays with 2 or more dimensions. */
public interface Dim2 : Dim3

/** Marker for arrays with 1 or more dimensions. */
public interface Dim1 : Dim2

/**
 * Returns specific [Dimension] by integer [dim].
 */
@Suppress("NOTHING_TO_INLINE", "UNCHECKED_CAST")
public inline fun <D : Dimension> dimensionOf(dim: Int): D = when (dim) {
    1 -> D1
    2 -> D2
    3 -> D3
    4 -> D4
    else -> DN(dim)
} as D

/**
 * Returns a [Dimension] singleton matching the reified type [D], using [dim] as a fallback rank for [DN].
 */
public inline fun <reified D : Dimension> dimensionClassOf(dim: Int = -1): D = when (D::class) {
    D1::class -> D1
    D2::class -> D2
    D3::class -> D3
    D4::class -> D4
    else -> DN(dim)
} as D

/**
 * Represents an N-dimensional rank (typically > 4, or when rank is unknown at compile time).
 *
 * Unlike [D1]–[D4], [DN] is a regular class that stores the actual rank in [d].
 *
 * @param d the number of dimensions.
 */
public class DN(override val d: Int) : Dimension, DimN {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other != null && this::class != other::class) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }
}

/**
 * Singleton representing 4-dimensional arrays. Use [D4] as both a type parameter and the companion instance.
 */
public sealed class D4(override val d: Int = 4) : Dimension, Dim4 {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other != null && this::class != other::class) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }

    public companion object : D4()
}

/**
 * Singleton representing 3-dimensional arrays. Use [D3] as both a type parameter and the companion instance.
 */
public sealed class D3(override val d: Int = 3) : Dimension, Dim3 {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other != null && this::class != other::class) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }

    public companion object : D3()
}

/**
 * Singleton representing 2-dimensional arrays (matrices). Use [D2] as both a type parameter and the companion instance.
 */
public sealed class D2(override val d: Int = 2) : Dimension, Dim2 {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other != null && this::class != other::class) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }

    public companion object : D2()
}

/**
 * Singleton representing 1-dimensional arrays (vectors). Use [D1] as both a type parameter and the companion instance.
 */
public sealed class D1(override val d: Int = 1) : Dimension, Dim1 {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other != null && this::class != other::class) return false

        other as DN
        if (d != other.d) return false
        return true
    }

    override fun hashCode(): Int = 31 * d

    override fun toString(): String {
        return "dimension: $d"
    }

    public companion object : D1()
}
