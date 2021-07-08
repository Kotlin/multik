/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.cuda

import jcuda.jcublas.JCublas
import org.jetbrains.kotlinx.multik.api.*


public class CudaEngineProvider : EngineProvider {
    override fun getEngine(): Engine {
        return CudaEngine
    }
}

public object CudaEngine : Engine(), AutoCloseable {
    override val name: String
        get() = type.name

    override val type: EngineType
        get() = CudaEngineType

    init {
        JCublas.cublasInit()
    }

    override fun getMath(): Math {
        return CudaMath
    }

    override fun getLinAlg(): LinAlg {
        return CudaLinAlg
    }

    override fun getStatistics(): Statistics {
        return CudaStatistics
    }

    override fun close() {
        JCublas.cublasShutdown()
    }
}