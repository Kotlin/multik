package org.jetbrains.kotlinx.multik.jni.linalg

import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.CopyStrategy
import org.jetbrains.kotlinx.multik.ndarray.operations.isTransposed
import org.jetbrains.kotlinx.multik.ndarray.operations.toType

public object NativeLinAlgEx : LinAlgEx {
    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<Double, D2> =
        invCommon(mat.toType(CopyStrategy.MEANINGFUL))

    override fun invF(mat: MultiArray<Float, D2>): NDArray<Float, D2> =
        invCommon(if (mat.consistent) mat.copy() else mat.deepCopy())

    override fun <T : Complex> invC(mat: MultiArray<T, D2>): NDArray<T, D2> =
        invCommon(if (mat.consistent) mat.copy() else mat.deepCopy())

    private fun <T> invCommon(mat: MultiArray<T, D2>): NDArray<T, D2> {
        requireSquare(mat.shape)

        val info: Int = when (mat.dtype) {
            DataType.FloatDataType -> JniLinAlg.inv(mat.shape[0], mat.data.getFloatArray(), mat.strides[0])
            DataType.DoubleDataType -> JniLinAlg.inv(mat.shape[0], mat.data.getDoubleArray(), mat.strides[0])
            DataType.ComplexFloatDataType -> JniLinAlg.invC(mat.shape[0], mat.data.getFloatArray(), mat.strides[0])
            DataType.ComplexDoubleDataType -> JniLinAlg.invC(mat.shape[0], mat.data.getDoubleArray(), mat.strides[0])
            else -> throw UnsupportedOperationException()
        }

        when {
            info < 0 -> throw IllegalArgumentException("${-info} argument had illegal value. ")
            info > 0 -> throw Exception("U($info, $info) is exactly zero. Matrix is singular and its inverse could not be computed")
        }

        return mat as NDArray<T, D2>
    }

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<Double, D> =
        solveCommon(a.toType(CopyStrategy.MEANINGFUL), b.toType(CopyStrategy.MEANINGFUL))

    override fun <D : Dim2> solveF(a: MultiArray<Float, D2>, b: MultiArray<Float, D>): NDArray<Float, D> =
        solveCommon(a.deepCopy(), b.deepCopy())

    override fun <T : Complex, D : Dim2> solveC(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> =
        solveCommon(a.deepCopy(), b.deepCopy())

    private fun <T, D : Dim2> solveCommon(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        requireSquare(a.shape)
        require(a.shape[0] == b.shape[0]) {
            "The first dimensions of the ndarrays a and b must be match: ${a.shape[0]}(a.shape[0]) != ${b.shape[0]}(b.shape[0]"
        }

        val nhrs = b.shape.last()

        val info: Int = when (a.dtype) {
            DataType.FloatDataType -> JniLinAlg.solve(
                a.shape[0], nhrs, a.data.getFloatArray(), a.strides[0], b.data.getFloatArray(), b.strides[0]
            )
            DataType.DoubleDataType -> JniLinAlg.solve(
                a.shape[0], nhrs, a.data.getDoubleArray(), a.strides[0], b.data.getDoubleArray(), b.strides[0]
            )
            DataType.ComplexFloatDataType -> JniLinAlg.solveC(
                a.shape[0], nhrs, a.data.getFloatArray(), a.strides[0], b.data.getFloatArray(), b.strides[0]
            )
            DataType.ComplexDoubleDataType -> JniLinAlg.solveC(
                a.shape[0], nhrs, a.data.getDoubleArray(), a.strides[0], b.data.getDoubleArray(), b.strides[0]
            )
            else -> throw UnsupportedOperationException()
        }
        if (info > 0) {
            throw Exception("The diagonal element of the triangular factor of a is zero, so that A is singular. The solution could not be computed.")
        }

        return b as NDArray<T, D>
    }

    override fun <T : Number> dotMM(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        dotMMCommon(a, b)

    override fun <T : Complex> dotMMComplex(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> =
        dotMMCommon(a, b)

    private fun <T> dotMMCommon(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2> {
        requireDotShape(a.shape, b.shape)

        val shape = intArrayOf(a.shape[0], b.shape[1])
        val size = shape.reduce(Int::times)
        val m = a.shape[0]
        val k = a.shape[1]
        val n = b.shape[1]

        val transA = a.isTransposed()
        val transB = b.isTransposed()

        val aN = if (transA || a.strides[1] == 1) a else a.deepCopy()
        val bN = if (transB || b.strides[1] == 1) b else b.deepCopy()

        val lda = if (transA) m else aN.strides[0]
        val ldb = if (transB) k else bN.strides[0]

        val cView = initMemoryView<T>(size, a.dtype)

        when (a.dtype) {
            DataType.FloatDataType ->
                JniLinAlg.dotMM(
                    transA,
                    aN.offset,
                    aN.data.getFloatArray(),
                    m,
                    k,
                    lda,
                    transB,
                    b.offset,
                    bN.data.getFloatArray(),
                    n,
                    ldb,
                    cView.getFloatArray()
                )
            DataType.DoubleDataType ->
                JniLinAlg.dotMM(
                    transA,
                    aN.offset,
                    aN.data.getDoubleArray(),
                    m,
                    k,
                    lda,
                    transB,
                    b.offset,
                    bN.data.getDoubleArray(),
                    n,
                    ldb,
                    cView.getDoubleArray()
                )
            DataType.ComplexFloatDataType ->
                JniLinAlg.dotMMC(
                    transA,
                    aN.offset,
                    aN.data.getFloatArray(),
                    m,
                    k,
                    lda,
                    transB,
                    b.offset,
                    bN.data.getFloatArray(),
                    n,
                    ldb,
                    cView.getFloatArray()
                )
            DataType.ComplexDoubleDataType ->
                JniLinAlg.dotMMC(
                    transA,
                    aN.offset,
                    aN.data.getDoubleArray(),
                    m,
                    k,
                    lda,
                    transB,
                    b.offset,
                    bN.data.getDoubleArray(),
                    n,
                    ldb,
                    cView.getDoubleArray()
                )
            else -> throw UnsupportedOperationException()
        }

        return D2Array(cView, 0, shape, dim = D2)
    }

    override fun <T : Number> dotMV(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        dotMVCommon(a, b)

    override fun <T : Complex> dotMVComplex(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> =
        dotMVCommon(a, b)

    private fun <T> dotMVCommon(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1> {
        requireDotShape(a.shape, b.shape)

        val size = a.shape[0]
        val shape = intArrayOf(size)
        val m = a.shape[0]
        val n = a.shape[1]

        val transA = a.isTransposed()
        val aN = if (transA || a.strides[1] == 1) a else a.deepCopy()
        val lda = if (transA) m else aN.strides[0]

        val cView = initMemoryView<T>(size, a.dtype)

        when (a.dtype) {
            DataType.FloatDataType ->
                JniLinAlg.dotMV(
                    transA,
                    aN.offset,
                    aN.data.getFloatArray(),
                    m,
                    n,
                    lda,
                    b.data.getFloatArray(),
                    b.strides[0],
                    cView.getFloatArray()
                )
            DataType.DoubleDataType ->
                JniLinAlg.dotMV(
                    transA,
                    aN.offset,
                    aN.data.getDoubleArray(),
                    m,
                    n,
                    lda,
                    b.data.getDoubleArray(),
                    b.strides[0],
                    cView.getDoubleArray()
                )
            DataType.ComplexFloatDataType ->
                JniLinAlg.dotMVC(
                    transA,
                    aN.offset,
                    aN.data.getFloatArray(),
                    m,
                    n,
                    lda,
                    b.data.getFloatArray(),
                    b.strides[0],
                    cView.getFloatArray()
                )
            DataType.ComplexDoubleDataType ->
                JniLinAlg.dotMVC(
                    transA,
                    aN.offset,
                    aN.data.getDoubleArray(),
                    m,
                    n,
                    lda,
                    b.data.getDoubleArray(),
                    b.strides[0],
                    cView.getDoubleArray()
                )
            else -> throw UnsupportedOperationException()
        }

        return D1Array(cView, 0, shape, dim = D1)
    }

    override fun <T : Number> dotVV(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        require(a.size == b.size) { "Vector sizes don't match: ${a.size} != ${b.size}" }

        return when (a.dtype) {
            DataType.FloatDataType ->
                JniLinAlg.dotVV(a.size, a.data.getFloatArray(), a.strides[0], b.data.getFloatArray(), b.strides[0])
            DataType.DoubleDataType ->
                JniLinAlg.dotVV(a.size, a.data.getDoubleArray(), a.strides[0], b.data.getDoubleArray(), b.strides[0])
            else -> throw UnsupportedOperationException()
        } as T
    }

    override fun <T : Complex> dotVVComplex(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        require(a.size == b.size) { "Vector sizes don't match: ${a.size} != ${b.size}" }

        return when (a.dtype) {
            DataType.ComplexFloatDataType ->
                JniLinAlg.dotVVC(a.size, a.data.getFloatArray(), a.strides[0], b.data.getFloatArray(), b.strides[0])
            DataType.ComplexDoubleDataType ->
                JniLinAlg.dotVVC(a.size, a.data.getDoubleArray(), a.strides[0], b.data.getDoubleArray(), b.strides[0])
            else -> throw UnsupportedOperationException()
        } as T
    }
}

private fun requireDotShape(aShape: IntArray, bShape: IntArray) = require(aShape[1] == bShape[0]) {
    "Shapes mismatch: shapes " +
        "${aShape.joinToString(prefix = "(", postfix = ")")} and " +
        "${bShape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
        "${aShape[1]} (dim 1) != ${bShape[0]} (dim 0)"
}


private fun requireSquare(shape: IntArray) = require(shape[0] == shape[1]) {
    "Ndarray must be square: shape = ${shape.joinToString(",", "(", ")")}"
}