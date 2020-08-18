package org.jetbrains.multik

import org.jetbrains.multik.api.JvmMath
import org.jetbrains.multik.api.d2array
import org.jetbrains.multik.api.mk
import org.jetbrains.multik.core.D2Array
import org.jetbrains.multik.jni.NativeMath
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import java.util.concurrent.TimeUnit
import kotlin.random.Random

@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 5)
@Measurement(iterations = 5)
@Fork(value = 2, jvmArgsPrepend = ["-Djava.library.path=./build/libs"])
open class MaxBenchmark {
    @Param("10", "100", "1000")
    var size: Int = 0
    private lateinit var arg: D2Array<Double>
    private var result: Double = 0.0
    private lateinit var ran: Random

    @Setup
    fun generate() {
        ran = Random(1)
        arg = mk.d2array(size, size) { ran.nextDouble() }
    }

    @Benchmark
    fun maxJniBench(bh: Blackhole) {
        result = NativeMath.max(arg)
        bh.consume(result)
    }

    @Benchmark
    fun maxJvmBench(bh: Blackhole) {
        result = JvmMath.max(arg)
        bh.consume(result)
    }
}