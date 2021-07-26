package org.jetbrains.kotlinx.multik.cuda

import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.junit.BeforeClass
import org.junit.Test
import org.slf4j.simple.SimpleLogger

val MB = 1024 * 1024L

object CudaTest {
    private val logger = KotlinLogging.logger {}

    @Test
    fun testGCFreeing() {
        logger.info { "Test GC Freeing" }

        val heapFreeSize = Runtime.getRuntime().freeMemory() / MB
        logger.info {"heapFreeSize $heapFreeSize MB"}

        val size = 10 * MB / 4

        CudaEngine.runWithCuda {
            for (i in 0 until 10) {
                val arr = mk.empty<Float, D1>(size.toInt())

                GpuArray.getOrAlloc(arr, false)
            }
        }
    }

    @Test
    fun testCacheFreeing() {
        logger.info { "Test Cache Freeing" }
        val heapFreeSize = Runtime.getRuntime().freeMemory() / MB
        logger.info {"heapFreeSize $heapFreeSize MB"}

        val size = MB / 4

        CudaEngine.runWithCuda {
            val arr1 = mk.empty<Float, D1>(1000 * size.toInt())
            val arr2 = mk.empty<Float, D1>(1000 * size.toInt())
            val arr3 = mk.empty<Float, D1>(2000 * size.toInt())
            val arr4 = mk.empty<Float, D1>(2000 * size.toInt())

            GpuArray.getOrAlloc(arr1, false)
            GpuArray.getOrAlloc(arr2, false)
            GpuArray.getOrAlloc(arr3, false)
            GpuArray.getOrAlloc(arr4, false)
        }
    }
}

fun main() {
    System.setProperty(SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "TRACE");
    //System.setProperty(SimpleLogger.LOG_FILE_KEY, "CudaTest.log");

    CudaTest.testGCFreeing()

    CudaTest.testCacheFreeing()
}