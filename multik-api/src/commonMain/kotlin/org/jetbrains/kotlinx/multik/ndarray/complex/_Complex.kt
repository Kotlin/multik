package org.jetbrains.kotlinx.multik.ndarray.complex

public fun Number.toComplexFloat(): ComplexFloat = ComplexFloat(this.toFloat(), 0f)

public fun Number.toComplexDouble(): ComplexDouble = ComplexDouble(this.toDouble(), 0.0)