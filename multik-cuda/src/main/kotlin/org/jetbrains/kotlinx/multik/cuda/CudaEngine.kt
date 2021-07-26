/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasHandle
import mu.KotlinLogging
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.linalg.LinAlg

private val logger = KotlinLogging.logger {}

public class CudaEngineProvider : EngineProvider {
    override fun getEngine(): Engine {
        return CudaEngine
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
        if (contextHandle != null) {
            logger.warn { "Trying to initialize the CudaEngine when it is already initialized" }
            return
        }

        logger.info { "Initializing cuda engine" }
        contextHandle = cublasHandle()
        JCublas2.cublasCreate(contextHandle)
    }

    public fun deinitCuda() {
        if (contextHandle == null) {
            logger.warn { "Trying to deinitialize the CudaEngine when it is already deinitialized" }
            return
        }

        logger.info { "Deinitializing cuda engine" }
        JCublas2.cublasDestroy(contextHandle)
        contextHandle = null
    }

    internal var contextHandle: cublasHandle? = null
        private set
}