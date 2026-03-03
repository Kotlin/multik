package org.jetbrains.kotlinx.multik.openblas.stat


internal expect object JniStat {
    fun median(arr: Any, size: Int, dtype: Int): Double
}
