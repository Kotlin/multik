package org.jetbrains.kotlinx.multik.cuda.linalg

import jcuda.jcublas.JCublas2
import jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_DEFAULT
import jcuda.jcublas.cublasOperation
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.cuda.*
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.*

public object CudaLinAlgEx: LinAlgEx {
    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> {
        TODO("Not yet implemented")
    }

    override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> {
        TODO("Not yet implemented")
    }

    override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> {
        TODO("Not yet implemented")
    }

    override fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        TODO("Not yet implemented")
    }

    override fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        dotMM_MV(a, b)

    override fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        dotMM_MV(a, b)

    override fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        require(a.shape[0] == b.shape[0]) {
            "Shapes mismatch: shapes " +
                    "${a.shape.joinToString(prefix = "(", postfix = ")")} and " +
                    "${b.shape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
                    "${a.shape[0]} (dim 0) != ${b.shape[0]} (dim 0)"
        }

        if (!(a.dtype == DataType.DoubleDataType || a.dtype == DataType.FloatDataType)) {
            throw UnsupportedOperationException("Unsupported data type: ${a.dtype}")
        }

        val context = CudaEngine.getContext()

        val gA = context.cache.getOrAlloc(a)
        val gB = context.cache.getOrAlloc(b)

        context.cache.assertAllLoaded(gA, gB)

        val result = initMemoryView<T>(1, a.dtype)
        val resultPtr = result.getDataPointer()
        val type = a.dtype.getCudaType()

        checkResult(JCublas2.cublasDotEx(context.handle, a.shape[0], gA.deviceDataPtr, type, 1, gB.deviceDataPtr, type, 1, resultPtr, type, type))

        return result[0]
    }

    private fun <T : Number, D : Dim2> dotMM_MV(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        require(a.shape[1] == b.shape[0]) {
            "Shapes mismatch: shapes " +
                    "${a.shape.joinToString(prefix = "(", postfix = ")")} and " +
                    "${b.shape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
                    "${a.shape[1]} (dim 1) != ${b.shape[0]} (dim 0)"
        }

        if (!(a.dtype == DataType.DoubleDataType || a.dtype == DataType.FloatDataType)) {
            throw UnsupportedOperationException("Unsupported data type: ${a.dtype}")
        }

        val matrixMatrix = b.dim.d == 2

        val shape = if (matrixMatrix)
            intArrayOf(a.shape[0], b.shape[1])
        else
            intArrayOf(a.shape[0])

        val cSize = shape.reduce(Int::times)

        val context = CudaEngine.getContext()

        val gA = context.cache.getOrAlloc(a)
        val gB = context.cache.getOrAlloc(b)

        val (result, gC) = context.cache.alloc<T, D>(cSize, a.dtype, shape, b.dim)

        context.cache.assertAllLoaded(gA, gB, gC)

        val zeroPtr = a.dtype.getZeroPointer()
        val onePtr = a.dtype.getOnePointer()

        if (matrixMatrix) {
            val m = a.shape[0]
            val n = b.shape[1]
            val k = a.shape[1]

            val transA = if (gA.transposed) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N
            val transB = if (gB.transposed) cublasOperation.CUBLAS_OP_T else cublasOperation.CUBLAS_OP_N

            val lda = if (gA.transposed) m else k
            val ldb = if (gB.transposed) k else n

            val type = a.dtype.getCudaType()
            val computeType = a.dtype.getDefaultComputeType()

            // multiplication order is swapped because cublas uses column-major storage
            checkResult(JCublas2.cublasGemmEx_new(
                context.handle, transB, transA, n, m, k,
                onePtr, gB.deviceDataPtr, type, ldb, gA.deviceDataPtr, type, lda, zeroPtr, gC.deviceDataPtr, type, n,
                computeType, CUBLAS_GEMM_DEFAULT
            ))
        } else {
            val transA = if (gA.transposed) cublasOperation.CUBLAS_OP_N else cublasOperation.CUBLAS_OP_T // TODO: check transposed

            var (m, n) = a.shape
            if (!gA.transposed)
                m = n.also { n = m }

            val func = if (a.dtype == DataType.DoubleDataType) JCublas2::cublasDgemv else JCublas2::cublasSgemv

            checkResult(func(context.handle, transA, m, n, onePtr, gA.deviceDataPtr, m, gB.deviceDataPtr, 1, zeroPtr, gC.deviceDataPtr, 1))
        }

        return result
    }
}