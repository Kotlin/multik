package org.jetbrains.multik.ndarray.data

public class MultiIndexProgression(public val first: IntArray, public val last: IntArray, public val step: Int = 1) {

    init {
        if (step == 0) throw IllegalArgumentException("Step must be non-zero.")
        if (step == Int.MIN_VALUE) throw IllegalArgumentException("Step must be greater than Int.MIN_VALUE to avoid overflow on negation.")
        if (first.size != last.size) throw IllegalArgumentException("Sizes first and last must be identical.")
    }

    public operator fun iterator(): Iterator<IntArray> = MultiIndexIterator(first, last, step)

    override fun equals(other: Any?): Boolean =
        other is MultiIndexProgression && (first.contentEquals(other.first) && last.contentEquals(other.last))

    override fun hashCode(): Int {
        return (first + last).hashCode()
    }

    override fun toString(): String {
        return "${first.joinToString(prefix = "(", postfix = ")")}..${last.joinToString(prefix = "(", postfix = ")")}"
    }
}

internal class MultiIndexIterator(first: IntArray, last: IntArray, private val step: Int) : Iterator<IntArray> {
    private val finalElement: IntArray = IntArray(last.size) { last[it] - 1 }
    private var hasNext: Boolean = if (step > 0) {
        var ret: Boolean = true
        for (i in first.size - 1 downTo 0) {
            if (first[i] > last[i]) {
                ret = false
                break
            }
        }
        ret
    } else {
        var ret: Boolean = true
        for (i in first.size - 1 downTo 0) {
            if (first[i] < last[i]) {
                ret = false
                break
            }
        }
        ret
    }

    //todo (-1???)
    private val next = if (hasNext) first.apply { set(lastIndex, -1) } else finalElement

    override fun hasNext(): Boolean = hasNext

    override fun next(): IntArray {
        next += step
        if (next.contentEquals(finalElement)) {
            if (!hasNext) throw NoSuchElementException()
            hasNext = false
        }
        return next
    }

    private operator fun IntArray.plusAssign(value: Int) {
        for (i in this.size - 1 downTo 0) {
            val t = this[i] + value
            if (t > finalElement[i] && i != 0) {
                this[i] = 0
            } else {
                this[i] = t
                break
            }
        }
    }
}

public operator fun IntArray.rangeTo(other: IntArray): MultiIndexProgression {
    return MultiIndexProgression(this, other)
}

public infix fun MultiIndexProgression.step(step: Int): MultiIndexProgression {
    if (step <= 0) throw IllegalArgumentException("Step must be positive, was: $step.")
    return MultiIndexProgression(first, last, step)
}

public infix fun IntArray.downTo(to: IntArray): MultiIndexProgression {
    return MultiIndexProgression(this, to, -1)
}