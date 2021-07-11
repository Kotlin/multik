/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik.jvm

import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.set
import java.lang.ArithmeticException
import kotlin.math.absoluteValue
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.min

public object JvmLinAlg : LinAlg {

    override fun <T : Number> inv(mat: MultiArray<T, D2>): NDArray<T, D2> {
        TODO("Not yet implemented")
    }

    override fun <T : Number> pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2> {
        if (n == 0) return mk.identity(mat.shape[0], mat.dtype)

        return if (n % 2 == 0) {
            val tmp = pow(mat, n / 2)
            dot(tmp, tmp)
        } else {
            dot(mat, pow(mat, n - 1))
        }

    }



    override fun <T : Number> norm(mat: MultiArray<T, D2>, p: Int): Double {

        require(p > 0) { "power $p must be positive" }

        return when (mat.dtype) {
            DataType.DoubleDataType -> {
                norm(mat.data.getDoubleArray(), mat.offset, mat.strides, mat.shape[0], mat.shape[1], p, mat.consistent)
            }
            DataType.FloatDataType -> {
                norm(mat.data.getFloatArray(), mat.offset, mat.strides, mat.shape[0], mat.shape[1], p, mat.consistent)
            }
            else -> {
                normGeneral(mat.data, mat.offset, mat.strides, mat.shape[0], mat.shape[1], p, mat.consistent)
            }
        }

    }

    //----------------- start cases for norm method ----------------------
    private fun norm(
        mat: FloatArray, matOffset: Int, matStrides: IntArray,
        n: Int, m: Int, power: Int,
        consistent: Boolean
    ): Double {
        var result = 0.0

        val (matStride_0, matStride_1) = matStrides

        if (consistent) {
            for (element in mat) {
                result += (element.absoluteValue.toDouble()).pow(power)
            }
        } else {
            for (i in 0 until n) {
                val matInd = i * matStride_0 + matOffset
                for (k in 0 until m) {
                    val elementDoubleAbsValue = mat[matInd + k * matStride_1].absoluteValue.toDouble()
                    result += (elementDoubleAbsValue).pow(power)
                }
            }
        }

        return result.pow(1 / power.toDouble())
    }

    private fun norm(
        mat: DoubleArray, matOffset: Int, matStrides: IntArray,
        n: Int, m: Int, power: Int,
        consistent: Boolean
    ): Double {
        //most common case of matrix elements
        var result = 0.0

        val (matStride_0, matStride_1) = matStrides

        if (consistent) {
            result = mat.sumOf { abs(it).pow(power) }
        } else {
            for (i in 0 until n) {
                val matInd = i * matStride_0 + matOffset
                for (k in 0 until m) {
                    val elementDoubleAbsValue = abs(mat[matInd + k * matStride_1])
                    result += (elementDoubleAbsValue).pow(power)
                }
            }
        }

        return result.pow(1 / power.toDouble())
    }

    private fun <T : Number> normGeneral(
        mat: ImmutableMemoryView<T>, matOffset: Int, matStrides: IntArray,
        n: Int, m: Int, power: Int,
        consistent: Boolean
    ): Double {
        var result = 0.0

        val (matStride_0, matStride_1) = matStrides

        if (consistent) {
            result = mat.sumOf { abs(it.toDouble()).pow(power) }
        } else {
            for (i in 0 until n) {
                val matInd = i * matStride_0 + matOffset
                for (k in 0 until m) {
                    val elementDoubleAbsValue = abs(mat[matInd + k * matStride_1].toDouble())
                    result += (elementDoubleAbsValue).pow(power)
                }
            }
        }

        return result.pow(1 / power.toDouble())
    }
//----------------- end of cases for norm method ----------------------

//-------------------------------start of LU-everything----------------------------------
    /**
     * computes alpha * a * b + beta * c and stores result is c
     * @param alpha scalar
     * @param beta scalar
     * @param a matrix
     * @param b matrix
     * @param c matrix
     *
     * blas / lapack: dgemm
     */
    public fun gemm(a: D2Array<Double>, shifta0: Int, shifta1: Int, ma: Int, na: Int, alpha: Double,
                     b: D2Array<Double>, shiftb0: Int, shiftb1: Int, mb: Int, nb: Int,
                     c: D2Array<Double>, shiftc0: Int, shiftc1: Int, mc: Int, nc: Int, beta: Double) {
        if(mc <= 0 || nc <= 0)
            return

        if (alpha == 0.0) {
            if (beta == 0.0) {
                for (i in shiftc0 until shiftc0 + mc) {
                    for (j in shiftc1 until shiftc1 + nc) {
                        c[i, j] = 0.0
                    }
                }
                return
            }
            for (i in shiftc0 until shiftc0 + mc) {
                for (j in shiftc1 until shiftc1 + nc) {
                    c[i, j] *= beta
                }
            }
            return
        }

        for (j in 0 until nc) {
            if (beta == 0.0) {
                for (i in 0 until mc) {
                    c[i + shiftc0, j + shiftc1] = 0.0
                }
            } else {
                for (i in 0 until mc) {
                    c[i + shiftc0, j + shiftc1] *= beta
                }
            }
            for (l in 0 until na) {
                val temp = alpha * b[shiftb0 + l, shiftb1 + j]
                for (i in 0 until mc) {
                    c[shiftc0 + i, shiftc1 + j] += temp * a[shifta0 + i, shifta1 + l]
                }
            }
        }
    }

    /**
     * Solves ax=b equation where @param a lower triangular matrix with units on diagonal,
     * rewrite @param a with solution
     *
     * @param a has na rows and na columns
     * @param b has na rows and nb columns
     *
     * notice: intentionally there is no checks that a[i, i] == 1.0 and a[i, >i] == 0.0,
     * it is a contract of this method having no such checks
     *
     * lapack: dtrsm can do it (and have some extra options)
     * TODO: think of visibility modifier
     */
    fun solveTriangleInplace(a: D2Array<Double>, shifta0: Int, shifta1: Int, na: Int, b: D2Array<Double>, shiftb0: Int, shiftb1: Int, nb: Int) {
        for (i in 0 until na) {
            for (k in i+1 until na) {
                for (j in 0 until nb) {
                    // array getter have extra two bound checks, maybe better use b.data[...]
                    b[k + shiftb0, j + shiftb1] -= a[k + shifta0, i + shifta1] * b[i + shiftb0, j + shiftb1]
                }
            }
        }
    }


    /**
     *
     * computes an LU factorization of a matrix @param a
     * @return triple of matrices p, l, u. Where u is permutation matrix,
     * l lower triangular matrix with unit diagonal elements
     * u upper triangular matrix
     *
     *
     * lapack: dgetrf2
     */
    public fun PLUdecomposition2Inplace(a: D2Array<Double>, shifta0: Int, shifta1: Int, ma: Int, na: Int, rowPerm: D1Array<Int>) {
        // this is recursive function, position of current matrix we work with is
        // a[shifta0 until shifta0 + am, shifta1 until shifta1 + an]
//        println("start call")
//        println(a)
//        println("test a:\n ${testSquarePLU(a)}")
//        println("params: shifta0=$shifta0, shifta1=$shifta1, ma=$ma, na=$na")
        val n1 = min(na, ma) / 2
        val n2 = na - n1

        // the idea of an algorithm is represent matrix a as
        // a = [ a11 a12 ]
        //     [ a21 a22 ]
        // where aij -- block submatrices
        // a11.shape = (n1, n1) (others shapes can be calculated using this information)
        // then recursively apply to [ a11 ] and [ a22 ] parts combining results together
        //                           [ a21 ]

        // corner cases
        if(na == 0 || ma == 0) {
            return
        }
        if (ma == 1) {
            return //because [[1]] * a == a
        }
        if (na == 1) {
            var imax = 0
            var elemmax = a[shifta0, shifta1]
            for (i in 1 until ma) {
                if (abs(a[shifta0 + i, shifta1]) > abs(elemmax)) {
                    elemmax = a[shifta0 + i, shifta1]
                    imax = i
                }
            }

            if (elemmax != 0.0) {
                // pivoting
                a[shifta0, shifta1] = a[shifta0 + imax, shifta1].also {
                    a[shifta0 + imax, shifta1] = a[shifta0, shifta1]
                }
                rowPerm[shifta0] = shifta0 + imax

                for (i in 1 until ma) {
                    a[shifta0 + i, shifta1] /= elemmax
                }
            }
//            println("column case")
//            println(a)
//            println("test a:\n ${testSquarePLU(a)}")

            return
        }

        // apply recursively to [ a11 ]
        //                      [ a21 ]
        PLUdecomposition2Inplace(a, shifta0, shifta1, ma, n1, rowPerm)
//        println("out of recursive call")
//        println(a)
//        println("test a:\n ${testSquarePLU(a)}")

        // change [ a12 ]
        //        [ a22 ]
        for (i in 0 until min(ma, rowPerm.size - shifta0)) {
            if (rowPerm[i + shifta0] != i + shifta0) {
                for (j in n1 until na) {
                    a[shifta0 + i, shifta1 + j] = a[rowPerm[i + shifta0], shifta1 + j].also {
                        a[rowPerm[i + shifta0], shifta1 + j] = a[shifta0 + i, shifta1 + j]
                    }
                }
            }
        }
//        println("right column perm")
//        println(a)
//        println("test a:\n ${testSquarePLU(a)}")

        // adjust a12
        solveTriangleInplace(a, shifta0, shifta1, n1, a, shifta0, shifta1 + n1, n2)

//        println("solved shifta0=$shifta0, shifta1=$shifta1, na=$n1, shiftb0=$shifta0, shiftb1=${shifta1 + n1}, nb=$n2")
//        println(a)
//        println("test a:\n ${testSquarePLU(a)}")

        // update a22
        gemm(a, shifta0 + n1, shifta1 + 0, ma - n1, n1, -1.0,
             a, shifta0 + 0, shifta1 + n1, n1, n2,
             a, shifta0 + n1, shifta1 + n1, ma - n1, n2, 1.0)

//        println("update a22")
//        println(a)
//        println("test a:\n ${testSquarePLU(a)}")

        // factor a22
        PLUdecomposition2Inplace(a, shifta0 + n1, shifta1 + n1, ma - n1, n2, rowPerm)

//        println("factor a22")
//        println(a)
//        println("test a:\n ${testSquarePLU(a)}")

        // apply rowPerm to a21
        for (i in n1 until ma) {
            if(shifta0 + i >= rowPerm.size) {
                break
            }
            if (rowPerm[shifta0 + i] != shifta0 + i) {
                for (j in 0 until n1) {
                    a[shifta0 + i, shifta1 + j] = a[rowPerm[shifta0 + i], shifta1 + j].also {
                        a[rowPerm[shifta0 + i], shifta1 + j] = a[shifta0 + i, shifta1 + j]
                    }
                }
            }
        }

//        println("apply perm to a21")
//        println(a)
//        println("test a:\n ${testSquarePLU(a)}")


    }

    fun PLU(a: D2Array<Double>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>> {
        val _a = a.deepCopy()
        val perm = mk.d1array<Int>(min(_a.shape[0], _a.shape[1])){ 0 }
        for (i in perm.indices) perm[i] = i

        PLUdecomposition2Inplace(_a, 0, 0, _a.shape[0], _a.shape[1], perm)
        // since previous call _a contains answer

        val L = mk.d2array(_a.shape[0], min(_a.shape[0], _a.shape[1])) {0.0}
        val U = mk.d2array(min(_a.shape[0], _a.shape[1]), _a.shape[1]) {0.0}

        for (i in 0 until L.shape[1]) {
            L[i, i] = 1.0
            for (j in 0 until i) {
                L[i, j] = _a[i, j]
            }
        }
        for (i in L.shape[1] until L.shape[0]) {
            for (j in 0 until L.shape[1]) {
                L[i, j] = _a[i, j]
            }
        }

        for (i in 0 until U.shape[0]) {
            for (j in i until U.shape[1]) {
                U[i, j] = _a[i, j]
            }
        }


        val P = mk.identity<Double>(_a.shape[0])

        for (i in perm.indices.reversed()) {
            if(perm[i] != i) {
                P[i] = P[perm[i]].deepCopy().also { P[perm[i]] = P[i].deepCopy() }
            }
        }
        return Triple(P, L, U)
    }

    fun PLUCompressed(a: D2Array<Double>): Triple<D1Array<Int>, D2Array<Double>, D2Array<Double>> {
        val _a = a.deepCopy()
        val perm = mk.d1array<Int>(min(_a.shape[0], _a.shape[1])){ 0 }
        for (i in perm.indices) perm[i] = i

        PLUdecomposition2Inplace(_a, 0, 0, _a.shape[0], _a.shape[1], perm)
        // since previous call _a contains answer

        val L = mk.d2array(_a.shape[0], min(_a.shape[0], _a.shape[1])) {0.0}
        val U = mk.d2array(min(_a.shape[0], _a.shape[1]), _a.shape[1]) {0.0}

        for (i in 0 until L.shape[1]) {
            L[i, i] = 1.0
            for (j in 0 until i) {
                L[i, j] = _a[i, j]
            }
        }
        for (i in L.shape[1] until L.shape[0]) {
            for (j in 0 until L.shape[1]) {
                L[i, j] = _a[i, j]
            }
        }

        for (i in 0 until U.shape[0]) {
            for (j in i until U.shape[1]) {
                U[i, j] = _a[i, j]
            }
        }


        val P = mk.identity<Double>(_a.shape[0])

        return Triple(perm.deepCopy(), L, U)
    }


    //-------------------------------end of LU-everything----------------------------------



    //-------------------------------solve linear system-----------------------------------
    /**
     * solves a*x = b where a lower or upper triangle square matrix
     */
    public fun solveTriangle(a: D2Array<Double>, b: D2Array<Double>, isLowerTriangle: Boolean = true): D2Array<Double> {
        require(a.shape[1] == b.shape[0]) { "invalid arguments, a.shape[1] = ${a.shape[1]} != b.shape[0]=${b.shape[0]}" }
        require(a.shape[0] == a.shape[1]) { "a should be a square matrix, matrix with shape (${a.shape[0]}, ${a.shape[1]}) given" }
        val x = b.deepCopy()
        for (i in 0 until x.shape[0]) {
            for (j in 0 until x.shape[1]) {
                x[i, j] /= a[i, i]
            }
        }
        if (isLowerTriangle) {
            for (i in 0 until x.shape[0]) {
                for (k in i + 1 until x.shape[0]) {
                    for (j in 0 until x.shape[1]) {
                        x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                    }
                }
            }
        } else {
            for (i in x.shape[0] - 1 downTo 0) {
                for (k in i - 1 downTo 0) {
                    for (j in 0 until x.shape[1]) {
                        x[k, j] -= a[k, i] * x[i, j] / a[k, k]
                    }
                }
            }

        }
        return x
    }

    fun solveDouble(a: D2Array<Double>, b: D2Array<Double>, singularityErrorLevel: Double = 1e-7): D2Array<Double> {
        require(a.shape[0] == a.shape[1] && a.shape[1] == b.shape[0])
        val (P, L, U) = PLUCompressed(a)
        val _b = b.deepCopy()
        for (i in 0 until P.size) {
            if(P[i] != i) {
                _b[i] = _b[P[i]].deepCopy().also { _b[P[i]] = _b[i].deepCopy() }
            }
        }
        for (i in 0 until U.shape[0]) {
            if (abs(U[i, i]) < singularityErrorLevel) {
                throw ArithmeticException("matrix a is almost singular")
            }
        }

        return solveTriangle(U, solveTriangle(L, _b), false)
    }

    override fun <T : Number, D : Dim2> solve(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        TODO("Not yet implemented")
    }

    //--------------------------end of solve linear system-----------------------------------



    override fun <T : Number, D : Dim2> dot(a: MultiArray<T, D2>, b: MultiArray<T, D>): NDArray<T, D> {
        require(a.shape[1] == b.shape[0]) {
            "Shapes mismatch: shapes " +
                    "${a.shape.joinToString(prefix = "(", postfix = ")")} and " +
                    "${b.shape.joinToString(prefix = "(", postfix = ")")} not aligned: " +
                    "${a.shape[1]} (dim 1) != ${b.shape[0]} (dim 0)"
        }

        return if (b.dim.d == 2) {
            dotMatrix(a, b as D2Array<T>) as NDArray<T, D>
        } else {
            dotVector(a, b as D1Array<T>) as NDArray<T, D>
        }
    }

    private fun <T : Number> dotMatrix(a: MultiArray<T, D2>, b: MultiArray<T, D2>): D2Array<T> {
        val newShape = intArrayOf(a.shape[0], b.shape[1])
        return when (a.dtype) {
            DataType.FloatDataType -> {
                val ret = D2Array(MemoryViewFloatArray(FloatArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.FloatDataType, dim = D2)
                dotMatrix(a.data.getFloatArray(), a.offset, a.strides, b.data.getFloatArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getFloatArray(), ret.strides[0])
                ret
            }
            DataType.IntDataType -> {
                val ret = D2Array(MemoryViewIntArray(IntArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.IntDataType, dim = D2)
                dotMatrix(a.data.getIntArray(), a.offset, a.strides, b.data.getIntArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getIntArray(), ret.strides[0])
                ret
            }
            DataType.DoubleDataType -> {
                val ret = D2Array(MemoryViewDoubleArray(DoubleArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.DoubleDataType, dim = D2)
                dotMatrix(a.data.getDoubleArray(), a.offset, a.strides, b.data.getDoubleArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getDoubleArray(), ret.strides[0])
                ret
            }
            DataType.LongDataType -> {
                val ret = D2Array(MemoryViewLongArray(LongArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.LongDataType, dim = D2)
                dotMatrix(a.data.getLongArray(), a.offset, a.strides, b.data.getLongArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getLongArray(), ret.strides[0])
                ret
            }
            DataType.ShortDataType -> {
                val ret = D2Array(MemoryViewShortArray(ShortArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.ShortDataType, dim = D2)
                dotMatrix(a.data.getShortArray(), a.offset, a.strides, b.data.getShortArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getShortArray(), ret.strides[0])
                ret
            }
            DataType.ByteDataType -> {
                val ret = D2Array(MemoryViewByteArray(ByteArray(newShape[0] * newShape[1])), 0, newShape, dtype = DataType.ByteDataType, dim = D2)
                dotMatrix(a.data.getByteArray(), a.offset, a.strides, b.data.getByteArray(), b.offset, b.strides, newShape[0], newShape[1], a.shape[1], ret.data.getByteArray(), ret.strides[0])
                ret
            }
        } as D2Array<T>
    }

    private fun dotMatrix(
        left: FloatArray, leftOffset: Int, leftStrides: IntArray,
        right: FloatArray, rightOffset: Int, rightStrides: IntArray,
        n: Int, m: Int, t: Int, destination: FloatArray, dStrides: Int
    ): FloatArray {
        val (leftStride_0, leftStride_1) = leftStrides
        val (rightStride_0, rightStride_1) = rightStrides

        for (i in 0 until n) {
            val dInd = i * dStrides
            val lInd = i * leftStride_0 + leftOffset
            for (k in 0 until t) {
                val ceil = left[lInd + k * leftStride_1]
                val rInd = k * rightStride_0 + rightOffset
                for (j in 0 until m) {
                    destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
                }
            }
        }
        return destination
    }

    private fun dotMatrix(
        left: ByteArray, leftOffset: Int, leftStrides: IntArray,
        right: ByteArray, rightOffset: Int, rightStrides: IntArray,
        n: Int, m: Int, t: Int, destination: ByteArray, dStrides: Int
    ): ByteArray {
        val (leftStride_0, leftStride_1) = leftStrides
        val (rightStride_0, rightStride_1) = rightStrides

        for (i in 0 until n) {
            val dInd = i * dStrides
            val lInd = i * leftStride_0 + leftOffset
            for (k in 0 until t) {
                val ceil = left[lInd + k * leftStride_1]
                val rInd = k * rightStride_0 + rightOffset
                for (j in 0 until m) {
                    destination[dInd + j] = (destination[dInd + j] + ceil * right[rInd + j * rightStride_1]).toByte()
                }
            }
        }
        return destination
    }

    private fun dotMatrix(
        left: ShortArray, leftOffset: Int, leftStrides: IntArray,
        right: ShortArray, rightOffset: Int, rightStrides: IntArray,
        n: Int, m: Int, t: Int, destination: ShortArray, dStrides: Int
    ): ShortArray {
        val (leftStride_0, leftStride_1) = leftStrides
        val (rightStride_0, rightStride_1) = rightStrides

        for (i in 0 until n) {
            val dInd = i * dStrides
            val lInd = i * leftStride_0 + leftOffset
            for (k in 0 until t) {
                val ceil = left[lInd + k * leftStride_1]
                val rInd = k * rightStride_0 + rightOffset
                for (j in 0 until m) {
                    destination[dInd + j] = (destination[dInd + j] + ceil * right[rInd + j * rightStride_1]).toShort()
                }
            }
        }
        return destination
    }

    private fun dotMatrix(
        left: IntArray, leftOffset: Int, leftStrides: IntArray,
        right: IntArray, rightOffset: Int, rightStrides: IntArray,
        n: Int, m: Int, t: Int, destination: IntArray, dStrides: Int
    ): IntArray {
        val (leftStride_0, leftStride_1) = leftStrides
        val (rightStride_0, rightStride_1) = rightStrides

        for (i in 0 until n) {
            val dInd = i * dStrides
            val lInd = i * leftStride_0 + leftOffset
            for (k in 0 until t) {
                val ceil = left[lInd + k * leftStride_1]
                val rInd = k * rightStride_0 + rightOffset
                for (j in 0 until m) {
                    destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
                }
            }
        }
        return destination
    }

    private fun dotMatrix(
        left: LongArray, leftOffset: Int, leftStrides: IntArray,
        right: LongArray, rightOffset: Int, rightStrides: IntArray,
        n: Int, m: Int, t: Int, destination: LongArray, dStrides: Int
    ): LongArray {
        val (leftStride_0, leftStride_1) = leftStrides
        val (rightStride_0, rightStride_1) = rightStrides

        for (i in 0 until n) {
            val dInd = i * dStrides
            val lInd = i * leftStride_0 + leftOffset
            for (k in 0 until t) {
                val ceil = left[lInd + k * leftStride_1]
                val rInd = k * rightStride_0 + rightOffset
                for (j in 0 until m) {
                    destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
                }
            }
        }
        return destination
    }

    private fun dotMatrix(
        left: DoubleArray, leftOffset: Int, leftStrides: IntArray,
        right: DoubleArray, rightOffset: Int, rightStrides: IntArray,
        n: Int, m: Int, t: Int, destination: DoubleArray, dStrides: Int
    ): DoubleArray {
        val (leftStride_0, leftStride_1) = leftStrides
        val (rightStride_0, rightStride_1) = rightStrides

        for (i in 0 until n) {
            val dInd = i * dStrides
            val lInd = i * leftStride_0 + leftOffset
            for (k in 0 until t) {
                val ceil = left[lInd + k * leftStride_1]
                val rInd = k * rightStride_0 + rightOffset
                for (j in 0 until m) {
                    destination[dInd + j] += ceil * right[rInd + j * rightStride_1]
                }
            }
        }
        return destination
    }

    private fun <T : Number> dotVector(a: MultiArray<T, D2>, b: MultiArray<T, D1>): D1Array<T> {
        val newShape = intArrayOf(a.shape[0])

        return when (a.dtype) {
            DataType.FloatDataType -> {
                val ret = D1Array(
                    MemoryViewFloatArray(FloatArray(newShape[0])),
                    0,
                    newShape,
                    dtype = DataType.FloatDataType,
                    dim = D1
                )
                dotVector(a.data.getFloatArray(), a.offset, a.strides, b.data.getFloatArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getFloatArray())
                ret
            }
            DataType.IntDataType -> {
                val ret = D1Array(MemoryViewIntArray(IntArray(newShape[0])), 0, newShape, dtype = DataType.IntDataType, dim = D1)
                dotVector(a.data.getIntArray(), a.offset, a.strides, b.data.getIntArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getIntArray())
                ret
            }
            DataType.DoubleDataType -> {
                val ret = D1Array(MemoryViewDoubleArray(DoubleArray(newShape[0])), 0, newShape, dtype = DataType.DoubleDataType, dim = D1)
                dotVector(a.data.getDoubleArray(), a.offset, a.strides, b.data.getDoubleArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getDoubleArray())
                ret
            }
            DataType.LongDataType -> {
                val ret = D1Array(MemoryViewLongArray(LongArray(newShape[0])), 0, newShape, dtype = DataType.LongDataType, dim = D1)
                dotVector(a.data.getLongArray(), a.offset, a.strides, b.data.getLongArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getLongArray())
                ret
            }
            DataType.ShortDataType -> {
                val ret = D1Array(MemoryViewShortArray(ShortArray(newShape[0])), 0, newShape, dtype = DataType.ShortDataType, dim = D1)
                dotVector(a.data.getShortArray(), a.offset, a.strides, b.data.getShortArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getShortArray())
                ret
            }
            DataType.ByteDataType -> {
                val ret = D1Array(MemoryViewByteArray(ByteArray(newShape[0])), 0, newShape, dtype = DataType.ByteDataType, dim = D1)
                dotVector(a.data.getByteArray(), a.offset, a.strides, b.data.getByteArray(), b.offset, b.strides[0], newShape[0], b.shape[0], ret.data.getByteArray())
                ret
            }
        } as D1Array<T>
    }

    private fun dotVector(
        left: FloatArray, leftOffset: Int, leftStrides: IntArray,
        right: FloatArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: FloatArray
    ): FloatArray {
        val (lStride_0, lStride_1) = leftStrides
        for (i in 0 until n) {
            val lInd = i * lStride_0 + leftOffset
            for (j in 0 until m) {
                destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
            }
        }
        return destination
    }

    private fun dotVector(
        left: IntArray, leftOffset: Int, leftStrides: IntArray,
        right: IntArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: IntArray): IntArray {
        val (lStride_0, lStride_1) = leftStrides
        for (i in 0 until n) {
            val lInd = i * lStride_0 + leftOffset
            for (j in 0 until m) {
                destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
            }
        }
        return destination
    }

    private fun dotVector(
        left: DoubleArray, leftOffset: Int, leftStrides: IntArray,
        right: DoubleArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: DoubleArray): DoubleArray {
        val (lStride_0, lStride_1) = leftStrides
        for (i in 0 until n) {
            val lInd = i * lStride_0 + leftOffset
            for (j in 0 until m) {
                destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
            }
        }
        return destination
    }

    private fun dotVector(
        left: LongArray, leftOffset: Int, leftStrides: IntArray,
        right: LongArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: LongArray): LongArray {
        val (lStride_0, lStride_1) = leftStrides
        for (i in 0 until n) {
            val lInd = i * lStride_0 + leftOffset
            for (j in 0 until m) {
                destination[i] += left[lInd + j * lStride_1] * right[j * rStride + rightOffset]
            }
        }
        return destination
    }

    private fun dotVector(
        left: ShortArray, leftOffset: Int, leftStrides: IntArray,
        right: ShortArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: ShortArray
    ): ShortArray {
        val (lStride_0, lStride_1) = leftStrides
        for (i in 0 until n) {
            val lInd = i * lStride_0 + leftOffset
            for (j in 0 until m) {
                destination[i] =
                    (destination[i] + left[lInd + j * lStride_1] * right[j * rStride + rightOffset]).toShort()
            }
        }
        return destination
    }

    private fun dotVector(
        left: ByteArray, leftOffset: Int, leftStrides: IntArray,
        right: ByteArray, rightOffset: Int, rStride: Int,
        n: Int, m: Int, destination: ByteArray
    ): ByteArray {
        val (lStride_0, lStride_1) = leftStrides
        for (i in 0 until n) {
            val lInd = i * lStride_0 + leftOffset
            for (j in 0 until m) {
                destination[i] =
                    (destination[i] + left[lInd + j * lStride_1] * right[j * rStride + rightOffset]).toByte()
            }
        }
        return destination
    }

    override fun <T : Number> dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T {
        require(a.size == b.size) { "Sizes a and b don't match: a.size(${a.size}) != b.size(${b.size})" }
        return when (a.dtype) {
            DataType.FloatDataType -> {
                dotVecToVec(a.data.getFloatArray(), a.offset, a.strides[0], b.data.getFloatArray(), b.offset, b.strides[0], a.size)
            }
            DataType.IntDataType -> {
                dotVecToVec(a.data.getIntArray(), a.offset, a.strides[0], b.data.getIntArray(), b.offset, b.strides[0], a.size)
            }
            DataType.DoubleDataType -> {
                dotVecToVec(a.data.getDoubleArray(), a.offset, a.strides[0], b.data.getDoubleArray(), b.offset, b.strides[0], a.size)
            }
            DataType.LongDataType -> {
                dotVecToVec(a.data.getLongArray(), a.offset, a.strides[0], b.data.getLongArray(), b.offset, b.strides[0], a.size)
            }
            DataType.ShortDataType -> {
                dotVecToVec(a.data.getShortArray(), a.offset, a.strides[0], b.data.getShortArray(), b.offset, b.strides[0], a.size)
            }
            DataType.ByteDataType -> {
                dotVecToVec(a.data.getByteArray(), a.offset, a.strides[0], b.data.getByteArray(), b.offset, b.strides[0], a.size)
            }
        } as T
    }

    private fun dotVecToVec(
        left: FloatArray, leftOffset: Int, lStride: Int,
        right: FloatArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Float {
        var ret = 0f
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret
    }

    private fun dotVecToVec(
        left: IntArray, leftOffset: Int, lStride: Int,
        right: IntArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Int {
        var ret = 0
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret
    }

    private fun dotVecToVec(
        left: DoubleArray, leftOffset: Int, lStride: Int,
        right: DoubleArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Double {
        var ret = 0.0
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret
    }

    private fun dotVecToVec(
        left: LongArray, leftOffset: Int, lStride: Int,
        right: LongArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Long {
        var ret = 0L
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret
    }

    private fun dotVecToVec(
        left: ShortArray, leftOffset: Int, lStride: Int,
        right: ShortArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Short {
        var ret = 0
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret.toShort()
    }

    private fun dotVecToVec(
        left: ByteArray, leftOffset: Int, lStride: Int,
        right: ByteArray, rightOffset: Int, rStride: Int,
        n: Int
    ): Byte {
        var ret = 0
        for (i in 0 until n) {
            ret += left[leftOffset + lStride * i] * right[rightOffset + rStride * i]
        }
        return ret.toByte()
    }
}
