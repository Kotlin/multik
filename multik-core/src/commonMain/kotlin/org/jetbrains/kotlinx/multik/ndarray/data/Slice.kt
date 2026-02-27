package org.jetbrains.kotlinx.multik.ndarray.data

import kotlin.jvm.JvmInline

/**
 * Marker interface unifying [Slice] and [RInt] index types for use in multi-axis slicing.
 */
public interface Indexing

/**
 * Placeholder indicating that a [Slice] should start from the first element of the axis.
 *
 * @see [sl.first]
 */
public class SliceStartStub

/**
 * Placeholder indicating that a [Slice] should extend to the last element of the axis.
 *
 * @see [sl.last]
 */
public class SliceEndStub

/**
 * Alias for [Slice.Companion], enabling `sl.first`, `sl.last`, and `sl.bounds` syntax.
 */
public typealias sl = Slice.Companion

/**
 * A start-boundary placeholder. Use in `sl.first..5` to slice from the beginning of an axis.
 */
public val sl.first: SliceStartStub
    get() = SliceStartStub()

/**
 * An end-boundary placeholder. Use in `3..sl.last` to slice to the end of an axis.
 */
public val sl.last: SliceEndStub
    get() = SliceEndStub()

/**
 * Returns a [Slice] spanning the full axis (equivalent to `sl.first..sl.last`).
 */
public val sl.bounds: Slice
    get() = Slice(-1, -1, 1)

/**
 * Defines a range with a step for array slicing, similar to Python's `slice(start, stop, step)`.
 *
 * Internally, `-1` for [start] or [stop] acts as a boundary placeholder meaning "from the beginning"
 * or "to the end" of the axis respectively.
 *
 * ```
 * val s = Slice(1, 5, 2) // elements at indices 1, 3, 5
 * val a = mk.ndarray(mk[10, 20, 30, 40, 50, 60])
 * a[s] // [20, 40, 60]
 * ```
 *
 * @see [sl]
 * @see [RInt]
 */
public class Slice(start: Int, stop: Int, step: Int) : Indexing, ClosedRange<Int> {

    init {
        if (step == 0 && start != 0 && stop != 0) throw IllegalArgumentException("Step must be non-zero.")
        if (step == Int.MIN_VALUE) throw IllegalArgumentException("Step must be greater than Int.MIN_VALUE to avoid overflow on negation.")
    }

    public companion object;

    private var _start: Int = start
    private var _stop: Int = stop

    public val step: Int = if (step < 0) throw IllegalArgumentException("Step must be positive.") else step
    public override val start: Int get() = _start
    public val stop: Int get() = _stop


    override val endInclusive: Int
        get() = stop


    public operator fun rangeTo(step: RInt): Slice = Slice(_start, _stop, step.data)

    public operator fun rangeTo(step: Int): Slice = Slice(_start, _stop, step)

    override fun contains(value: Int): Boolean = value in start..stop

    override fun equals(other: Any?): Boolean =
        other is Slice && (isEmpty() && other.isEmpty() ||
                start == other.start && stop == other.stop && step == other.step)

    override fun hashCode(): Int =
        if (isEmpty()) -1 else (31 * start + stop + step)

    override fun toString(): String = "$start..$stop..$step"
}

/**
 * Converts this [IntRange] to a [Slice] with step 1.
 */
public fun IntRange.toSlice(): Slice = Slice(this.first, this.last, 1)

/**
 * Converts this [ClosedRange] to a [Slice] with step 1.
 *
 * Supports [Slice], [IntRange], or any other [ClosedRange]<[Int]>.
 *
 * @throws IllegalStateException if the range type is not [Slice] or [IntRange].
 */
public fun ClosedRange<Int>.toSlice(): Slice =
    when(this) {
        is Slice -> this
        is IntRange -> this.toSlice()
        else -> throw IllegalStateException("${this::class} not supported, please use Slice or IntRange.")
    }

/**
 * Wraps this [Int] as an [RInt] for use in slice expressions.
 *
 * The `.r` suffix disambiguates indexing from Kotlin's built-in `rangeTo` operator,
 * enabling expressions like `0.r..5` to produce a [Slice].
 */
public val Int.r: RInt get() = RInt(this)

/**
 * An integer wrapper that implements [Indexing] and provides [Slice]-producing range operators.
 *
 * Use [Int.r] to create instances. [RInt] is needed because Kotlin's built-in `Int.rangeTo(Int)`
 * returns an [IntRange], not a [Slice]. Wrapping one operand as [RInt] selects the [Slice]-producing
 * overload instead.
 *
 * ```
 * val s = 1.r..5..2 // Slice(1, 5, 2)
 * ```
 */
@JvmInline
public value class RInt(internal val data: Int) : Indexing {

    public operator fun plus(r: RInt): RInt = RInt(this.data + r.data)
    public operator fun minus(r: RInt): RInt = RInt(this.data - r.data)
    public operator fun times(r: RInt): RInt = RInt(this.data * r.data)
    public operator fun div(r: RInt): RInt = RInt(this.data / r.data)

    public operator fun rangeTo(that: RInt): Slice = Slice(data, that.data, 1)
    public operator fun rangeTo(that: Int): Slice = Slice(data, that, 1)

    public infix fun until(that: RInt): Slice = Slice(data, that.data - 1, 1)

    public infix fun until(that: Int): Slice = Slice(data, that - 1, 1)
}

/**
 * Creates a [Slice] from the beginning of an axis to [that] (inclusive) with step 1.
 */
public operator fun SliceStartStub.rangeTo(that: Int): Slice = Slice(-1, that, 1)

/**
 * Creates a [Slice] from [this] to the end of an axis with step 1.
 */
public operator fun Int.rangeTo(that: SliceEndStub): Slice = Slice(this, -1, 1)

/**
 * Creates a [Slice] from [this] to [that]`.data` with step 1.
 */
public operator fun Int.rangeTo(that: RInt): Slice = Slice(this, that.data, 1)

/**
 * Creates a [Slice] from this [IntRange] with the specified [step].
 */
public operator fun IntRange.rangeTo(step: Int): Slice {
    return Slice(this.first, this.last, step)
}
