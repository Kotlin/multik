/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasHandle
import jcuda.runtime.JCuda
import jcuda.runtime.cudaStream_t
import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg
import org.jetbrains.kotlinx.multik.cuda.linalg.CudaLinAlg

private val logger = KotlinLogging.logger {}

public class CudaEngineProvider : EngineProvider {
    override fun getEngine(): Engine {
        return CudaEngine
    }
}

internal class CudaContext {
    val handle = cublasHandle()
    val cache = GpuCache()
    private val stream = cudaStream_t()

    init {
        checkResult(JCublas2.cublasCreate(handle))
        checkResult(JCuda.cudaStreamCreate(stream))

        checkResult(JCublas2.cublasSetStream(handle, stream))
    }

    fun deinit() {
        checkResult(JCublas2.cublasDestroy(handle))
        checkResult(JCuda.cudaStreamDestroy(stream))
    }
}

public object CudaEngine : Engine() {
    override val name: String
        get() = type.name

    override val type: EngineType
        get() = CudaEngineType

    override fun getMath(): Math {
        return CudaMath
    }

    override fun getLinAlg(): LinAlg {
        return CudaLinAlg
    }

    override fun getStatistics(): Statistics {
        return CudaStatistics
    }

    public fun runWithCuda(block: () -> Unit) {
        initCuda()
        block()
        deinitCuda()
    }

    public fun initCuda() {
        if (context.get() != null) {
            logger.warn { "Trying to initialize the CudaEngine when it is already initialized" }
            return
        }

        logger.info { "Initializing cuda engine" }
        context.set(CudaContext())
    }

    public fun deinitCuda() {
        if (context.get() == null) {
            logger.warn { "Trying to deinitialize the CudaEngine when it is not initialized" }
            return
        }

        logger.info { "Deinitializing cuda engine" }

        cacheCleanup(CleanupMode.FULL)

        context.get()!!.deinit()
        context.set(null)
    }

    public enum class CleanupMode {
        // Fully cleans the cache
        FULL,

        // Cleans the data that was garbage collected
        GARBAGE
    }

    public fun cacheCleanup(mode: CleanupMode = CleanupMode.GARBAGE) {
        logger.debug { "Cleaning cache. Mode: $mode" }

        when (mode) {
            CleanupMode.GARBAGE -> getContext().cache.garbageCleanup()
            CleanupMode.FULL -> getContext().cache.fullCleanup()
        }
    }

    private var context: ThreadLocal<CudaContext?> = ThreadLocal()

    internal fun getContext(): CudaContext =
        context.get() ?: throw IllegalStateException("Cuda is not initialized")
}