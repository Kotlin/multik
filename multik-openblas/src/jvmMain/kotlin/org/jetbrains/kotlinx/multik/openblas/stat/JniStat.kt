package org.jetbrains.kotlinx.multik.openblas.stat

internal actual object JniStat {
    actual external fun median(arr: Any, size: Int, dtype: Int): Double
}
