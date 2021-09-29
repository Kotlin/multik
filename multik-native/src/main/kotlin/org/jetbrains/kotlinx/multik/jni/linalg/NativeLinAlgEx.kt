package org.jetbrains.kotlinx.multik.jni.linalg

import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.empty
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.LinAlgEx
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.jni.NativeEngine
import org.jetbrains.kotlinx.multik.ndarray.complex.Complex
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.CopyStrategy
import org.jetbrains.kotlinx.multik.ndarray.operations.isTransposed
import org.jetbrains.kotlinx.multik.ndarray.operations.toType
import kotlin.math.min

public object NativeLinAlgEx : LinAlgEx {

    init {
        NativeEngine
    }

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

        val nhrs = if (b.dim.d == 1) 1 else b.shape.last()

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

    override fun <T : Number> qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>> =
        qrCommon(mat, DataType.DoubleDataType)

    override fun qrF(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>> =
        qrCommon(mat, DataType.FloatDataType)

    override fun <T : Complex> qrC(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>> =
        qrCommon(mat, mat.dtype)

    private fun <T, O: Any> qrCommon(mat: MultiArray<T, D2>, retDType: DataType): Pair<D2Array<O>, D2Array<O>> {
        val (m, n) = mat.shape
        val mn = min(m, n)
        val q = mat.toType<T, O, D2>(retDType, CopyStrategy.MEANINGFUL)
        val r = mk.empty<O, D2>(intArrayOf(mn, n), q.dtype)

        val info: Int = when (retDType) {
            DataType.FloatDataType -> JniLinAlg.qr(m, n, q.data.getFloatArray(), q.strides[0], r.data.getFloatArray())
            DataType.DoubleDataType -> JniLinAlg.qr(m, n, q.data.getDoubleArray(), q.strides[0], r.data.getDoubleArray())
            DataType.ComplexFloatDataType -> JniLinAlg.qrC(m, n, q.data.getFloatArray(), q.strides[0], r.data.getFloatArray())
            DataType.ComplexDoubleDataType -> JniLinAlg.qrC(m, n, q.data.getDoubleArray(), q.strides[0], r.data.getDoubleArray())
            else -> throw UnsupportedOperationException()
        }

        if (info < 0) throw IllegalArgumentException("${-info} argument had illegal value. ")

        // TODO internal copyOf(end: Int)
        return Pair(q[Slice.bounds, 0..mn].deepCopy() as D2Array<O>, r)
    }

    override fun <T : Number> plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> =
        pluCommon(mat, DataType.DoubleDataType, 1.0)

    override fun pluF(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>> =
        pluCommon(mat, DataType.FloatDataType, 1f)

    override fun <T : Complex> pluC(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>> =
        when (mat.dtype) {
            DataType.ComplexFloatDataType -> pluCommon(mat, mat.dtype, ComplexFloat.one as T)
            DataType.ComplexDoubleDataType -> pluCommon(mat, mat.dtype, ComplexDouble.one as T)
            else -> throw UnsupportedOperationException()
        }

//    override fun <T : Number> eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>> =
//        eigCommon<Double, ComplexDouble>(mat.toType(CopyStrategy.MEANINGFUL), true)
//            as Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>>

//    override fun eigF(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>> =
//        eigCommon<Float, ComplexFloat>(mat.deepCopy(), true)
//            as Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>>

//    override fun <T : Complex> eigC(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>> =
//        eigCommon<T, T>(mat.deepCopy(), true) as Pair<D1Array<T>, D2Array<T>>

    override fun <T : Number> eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble> =
        TODO("Not yet implemented")
//        eigCommon<Double, ComplexDouble>(mat.toType(CopyStrategy.MEANINGFUL), false).first

    override fun eigValsF(mat: MultiArray<Float, D2>): D1Array<ComplexFloat> =
        TODO("Not yet implemented")
//        eigCommon<Float, ComplexFloat>(mat.deepCopy(), false).first

    override fun <T : Complex> eigValsC(mat: MultiArray<T, D2>): D1Array<T> =
        TODO("Not yet implemented")
//        eigCommon<T, T>(mat.deepCopy(), false).first

    private fun <T, O : Complex> eigCommon(mat: MultiArray<T, D2>, computeVectors: Boolean): Pair<D1Array<O>, D2Array<O>?> {
        requireSquare(mat.shape)

        val n = mat.shape.first()
        val computeV = if (computeVectors) 'V' else 'N'
        val w: D1Array<O>
        val v: D2Array<O>?

        val info = when (mat.dtype) {
            DataType.FloatDataType -> {
                val wr = FloatArray(n)
                val wi = FloatArray(n)
                v = if (computeVectors) mk.empty(intArrayOf(n, n), DataType.ComplexFloatDataType) else null
                val i = JniLinAlg.eig(n, mat.data.getFloatArray(), wr, wi, computeV, v?.data?.getFloatArray())
                w = mk.d1array(n) { ComplexFloat(wr[it], wi[it]) } as D1Array<O>
                i
            }
            DataType.DoubleDataType -> {
                val wr = DoubleArray(n)
                val wi = DoubleArray(n)
                v = if (computeVectors) mk.empty(intArrayOf(n, n), DataType.ComplexDoubleDataType) else null
                val i = JniLinAlg.eig(n, mat.data.getDoubleArray(), wr, wi, computeV, v?.data?.getDoubleArray())
                w = mk.d1array(n) { ComplexDouble(wr[it], wi[it]) } as D1Array<O>
                i
            }
            DataType.ComplexFloatDataType -> {
                w = mk.empty(intArrayOf(n), DataType.ComplexFloatDataType)
                v = if (computeVectors) mk.empty(intArrayOf(n, n), DataType.ComplexFloatDataType) else null
                JniLinAlg.eig(n, mat.data.getFloatArray(), w.data.getFloatArray(), computeV, v?.data?.getFloatArray())
            }
            DataType.ComplexDoubleDataType -> {
                w = mk.empty(intArrayOf(n), DataType.ComplexDoubleDataType)
                v = if (computeVectors) mk.empty(intArrayOf(n, n), DataType.ComplexDoubleDataType) else null
                JniLinAlg.eig(n, mat.data.getDoubleArray(), w.data.getDoubleArray(), computeV, v?.data?.getDoubleArray())
            }
            else -> throw UnsupportedOperationException()
        }

        when {
            info < 0 -> throw IllegalArgumentException("The ${-info}-th argument had an illegal value")
            info > 0 -> throw Exception("Failed to compute all the eigenvalues.")
        }

        return Pair(w, v)
    }

    private fun <T, O : Any> pluCommon(mat: MultiArray<T, D2>, dtype: DataType, one: O): Triple<D2Array<O>, D2Array<O>, D2Array<O>> {
        val (m, n) = mat.shape
        val mn = min(m, n)

        val ipiv = IntArray(mn)
        val a = mat.toType<T, O, D2>(dtype, CopyStrategy.MEANINGFUL)

        val info: Int = when (dtype) {
            DataType.FloatDataType -> JniLinAlg.plu(m, n, a.data.getFloatArray(), mat.strides[0], ipiv)
            DataType.DoubleDataType -> JniLinAlg.plu(m, n, a.data.getDoubleArray(), mat.strides[0], ipiv)
            DataType.ComplexFloatDataType -> JniLinAlg.pluC(m, n, a.data.getFloatArray(), mat.strides[0], ipiv)
            DataType.ComplexDoubleDataType -> JniLinAlg.pluC(m, n, a.data.getDoubleArray(), mat.strides[0], ipiv)
            else -> throw UnsupportedOperationException()
        }

        if (info < 0) throw IllegalArgumentException("${-info} argument had illegal value. ")

        val P = mk.identity<O>(m, dtype)
        val L = mk.empty<O, D2>(intArrayOf(m, mn), dtype)
        val U = mk.empty<O, D2>(intArrayOf(mn, n), dtype)

        for (i in (ipiv.size - 1) downTo 0) {
            val ip = ipiv[i] - 1
            if (ip != 0) {
                for (k in 0 until P.shape[1]) {
                    P[i, k] = P[ip, k].also { P[ip, k] = P[i, k] }
                }
            }
        }

        for (i in 0 until m) {
            for (j in 0 until n) {
                when {
                    i == j -> {
                        U[i, j] = a[i, j]
                        L[i, j] = one
                    }
                    i < j && i < mn -> U[i, j] = a[i, j]
                    i > j && j < mn -> L[i, j] = a[i, j]
                }
            }
        }


        return Triple(P, L, U)
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
                    transA, aN.offset, aN.data.getFloatArray(), m, k, lda,
                    transB, b.offset, bN.data.getFloatArray(), n, ldb, cView.getFloatArray()
                )
            DataType.DoubleDataType ->
                JniLinAlg.dotMM(
                    transA, aN.offset, aN.data.getDoubleArray(), m, k, lda,
                    transB, b.offset, bN.data.getDoubleArray(), n, ldb, cView.getDoubleArray()
                )
            DataType.ComplexFloatDataType ->
                JniLinAlg.dotMMC(
                    transA, aN.offset, aN.data.getFloatArray(), m, k, lda,
                    transB, b.offset, bN.data.getFloatArray(), n, ldb, cView.getFloatArray()
                )
            DataType.ComplexDoubleDataType ->
                JniLinAlg.dotMMC(
                    transA, aN.offset, aN.data.getDoubleArray(), m, k, lda,
                    transB, b.offset, bN.data.getDoubleArray(), n, ldb, cView.getDoubleArray()
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
                    transA, aN.offset, aN.data.getFloatArray(), m, n, lda,
                    b.data.getFloatArray(), b.strides[0], cView.getFloatArray()
                )
            DataType.DoubleDataType ->
                JniLinAlg.dotMV(
                    transA, aN.offset, aN.data.getDoubleArray(), m, n, lda,
                    b.data.getDoubleArray(), b.strides[0], cView.getDoubleArray()
                )
            DataType.ComplexFloatDataType ->
                JniLinAlg.dotMVC(
                    transA, aN.offset, aN.data.getFloatArray(), m, n, lda,
                    b.data.getFloatArray(), b.strides[0], cView.getFloatArray()
                )
            DataType.ComplexDoubleDataType ->
                JniLinAlg.dotMVC(
                    transA, aN.offset, aN.data.getDoubleArray(), m, n, lda,
                    b.data.getDoubleArray(), b.strides[0], cView.getDoubleArray()
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