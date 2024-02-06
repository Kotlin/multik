package org.jetbrains.kotlinx.multik

import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDoubleArray
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloatArray
import org.jetbrains.kotlinx.multik.ndarray.complex.joinToString
import kotlin.test.assertTrue

infix fun ByteArray.shouldBe(expected: ByteArray) {
    assertTrue(
        "Expected <${
            expected.joinToString(prefix = "[", postfix = "]")
        }>, actual <${this.joinToString(prefix = "[", postfix = "]")}>."
    ) { this.contentEquals(expected) }
}

infix fun ShortArray.shouldBe(expected: ShortArray) {
    assertTrue(
        "Expected <${
            expected.joinToString(prefix = "[", postfix = "]")
        }>, actual <${this.joinToString(prefix = "[", postfix = "]")}>."
    ) { this.contentEquals(expected) }
}

infix fun IntArray.shouldBe(expected: IntArray) {
    assertTrue(
        "Expected <${
            expected.joinToString(prefix = "[", postfix = "]")
        }>, actual <${this.joinToString(prefix = "[", postfix = "]")}>."
    ) { this.contentEquals(expected) }
}

infix fun LongArray.shouldBe(expected: LongArray) {
    assertTrue(
        "Expected <${
            expected.joinToString(prefix = "[", postfix = "]")
        }>, actual <${this.joinToString(prefix = "[", postfix = "]")}>."
    ) { this.contentEquals(expected) }
}

infix fun FloatArray.shouldBe(expected: FloatArray) {
    assertTrue(
        "Expected <${
            expected.joinToString(prefix = "[", postfix = "]")
        }>, actual <${this.joinToString(prefix = "[", postfix = "]")}>."
    ) { this.contentEquals(expected) }
}

infix fun DoubleArray.shouldBe(expected: DoubleArray) {
    assertTrue(
        "Expected <${
            expected.joinToString(prefix = "[", postfix = "]")
        }>, actual <${this.joinToString(prefix = "[", postfix = "]")}>."
    ) { this.contentEquals(expected) }
}

infix fun ComplexFloatArray.shouldBe(expected: ComplexFloatArray) {
    assertTrue(
        "Expected <${
            expected.joinToString(prefix = "[", postfix = "]")
        }>, actual <${this.joinToString(prefix = "[", postfix = "]")}>."
    ) { this == expected }
}

infix fun ComplexDoubleArray.shouldBe(expected: ComplexDoubleArray) {
    assertTrue(
        "Expected <${
            expected.joinToString(prefix = "[", postfix = "]")
        }>, actual <${this.joinToString(prefix = "[", postfix = "]")}>."
    ) { this == expected }
}
