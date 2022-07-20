/*
 * Copyright 2020-2022 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.ndarray.data

import kotlin.jvm.JvmInline

/**
 * Marker class. Serves to share slice and simple indexes.
 */
public interface Indexing

/**
 * Convenient Slice for start stub.
 */
public class SliceStartStub

/**
 * Convenient Slice for stop stub.
 */
public class SliceEndStub

/**
 * Alias name for Slice.
 */
public typealias sl = Slice.Companion

/**
 * Stub start.
 */
public val sl.first: SliceStartStub
    get() = SliceStartStub()

/**
 * Stub stop.
 */
public val sl.last: SliceEndStub
    get() = SliceEndStub()

/**
 * Returns Slice with stubs the start and the stop.
 */
public val sl.bounds: Slice
    get() = Slice(-1, -1, 1)

/**
 * Slice class. An analogue of slices in python.
 */
public class Slice(start: Int, stop: Int, step: Int) : Indexing, ClosedRange<Int> {

    init {
        if (step == 0 && start != 0 && stop != 0) throw IllegalArgumentException("Step must be non-zero.")
        if (step == Int.MIN_VALUE) throw kotlin.IllegalArgumentException("Step must be greater than Int.MIN_VALUE to avoid overflow on negation.")
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
 * Returns Slice containing the first, the last with a step of 1.
 */
public fun IntRange.toSlice(): Slice = Slice(this.first, this.last, 1)

/**
 * Returns Slice containing the first, the last with a step of 1.
 */
public fun ClosedRange<Int>.toSlice(): Slice =
    when(this) {
        is Slice -> this
        is IntRange -> this.toSlice()
        else -> throw IllegalStateException("${this::class} not supported, please use Slice or IntRange.")
    }

/**
 * Returns [RInt].
 */
public val Int.r: RInt get() = RInt(this)

/**
 * Helper class for indexing. Since the standard rangeTo overrides the rangeTo for slices.
 */
@JvmInline
public value class RInt(internal val data: Int) : Indexing {

    public operator fun plus(r: RInt): RInt = RInt(this.data + r.data)
    public operator fun minus(r: RInt): RInt = RInt(this.data - r.data)
    public operator fun times(r: RInt): RInt = RInt(this.data * r.data)
    public operator fun div(r: RInt): RInt = RInt(this.data / r.data)

    public operator fun rangeTo(that: RInt): Slice = Slice(data, that.data, 1)
    public operator fun rangeTo(that: Int): Slice = Slice(data, that, 1)
}

/**
 * Returns Slice with stub of the start.
 */
public operator fun SliceStartStub.rangeTo(that: Int): Slice = Slice(-1, that, 1)

/**
 * Returns Slice with stub of the stop.
 */
public operator fun Int.rangeTo(that: SliceEndStub): Slice = Slice(this, -1, 1)

/**
 * Returns Slice where stop from RInt.
 */
public operator fun Int.rangeTo(that: RInt): Slice = Slice(this, that.data, 1)

/**
 * Returns a slice at a specified [step].
 */
public operator fun IntRange.rangeTo(step: Int): Slice {
    return Slice(this.first, this.last, step)
}
