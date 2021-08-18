/*
 * Copyright 2020-2021 JetBrains s.r.o. Use of this source code is governed by the Apache 2.0 license.
 */

package org.jetbrains.kotlinx.multik_jvm.linAlg


import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.jvm.eigenvalues
import org.jetbrains.kotlinx.multik.jvm.linalg.*
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlg.dot
import org.jetbrains.kotlinx.multik.jvm.linalg.JvmLinAlgEx.solve
import org.jetbrains.kotlinx.multik.jvm.qrComplexDouble
import org.jetbrains.kotlinx.multik.jvm.schurDecomposition
import org.jetbrains.kotlinx.multik.jvm.upperHessenberg
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random
import kotlin.test.*

class JvmLinAlgTest {

    @Test
    fun `test of norm function with p=1`() {
        val d2arrayDouble1 = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
        val d2arrayDouble2 = mk.ndarray(mk[mk[-1.0, -2.0], mk[-3.0, -4.0]])

        assertEquals(10.0, mk.linalg.norm(d2arrayDouble1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayDouble2, 1))

        val d2arrayShort1 = mk.ndarray(mk[mk[1.toShort(), 2.toShort()], mk[3.toShort(), 4.toShort()]])
        val d2arrayShort2 = mk.ndarray(mk[mk[(-1).toShort(), (-2).toShort()], mk[(-3).toShort(), (-4).toShort()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayShort1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayShort2, 1))

        val d2arrayInt1 = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val d2arrayInt2 = mk.ndarray(mk[mk[-1, -2], mk[-3, -4]])

        assertEquals(10.0, mk.linalg.norm(d2arrayInt1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayInt2, 1))

        val d2arrayByte1 = mk.ndarray(mk[mk[1.toByte(), 2.toByte()], mk[3.toByte(), 4.toByte()]])
        val d2arrayByte2 = mk.ndarray(mk[mk[(-1).toByte(), (-2).toByte()], mk[(-3).toByte(), (-4).toByte()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayByte1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayByte2, 1))

        val d2arrayFloat1 = mk.ndarray(mk[mk[1.0.toFloat(), 2.0.toFloat()], mk[3.0.toFloat(), 4.0.toFloat()]])
        val d2arrayFloat2 =
            mk.ndarray(mk[mk[(-1.0).toFloat(), (-2.0).toFloat()], mk[(-3.0).toFloat(), (-4.0).toFloat()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayFloat1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayFloat2, 1))

        val d2arrayLong1 = mk.ndarray(mk[mk[1.toLong(), 2.toLong()], mk[3.toLong(), 4.toLong()]])
        val d2arrayLong2 = mk.ndarray(mk[mk[(-1).toLong(), (-2).toLong()], mk[(-3).toLong(), (-4).toLong()]])

        assertEquals(10.0, mk.linalg.norm(d2arrayLong1, 1))
        assertEquals(10.0, mk.linalg.norm(d2arrayLong2, 1))

    }

    @Test
    fun `test plu`() {
        val procedurePrecision = 1e-5
        val rnd = Random(424242)

        val iters = 1000
        val sideFrom = 1
        val sideUntil = 100
        for (all in 0 until iters) {
            val m = rnd.nextInt(sideFrom, sideUntil)
            val n = rnd.nextInt(sideFrom, sideUntil)

            val a = mk.d2array<Double>(m, n) { rnd.nextDouble() }

            val (P, L, U) = plu(a)
            assertTriangular(L, isLowerTriangular = true, requireUnitsOnDiagonal = true)
            assertTriangular(U, isLowerTriangular = false, requireUnitsOnDiagonal = false)
            assertClose(dot(P, dot(L, U)), a, procedurePrecision)
        }
    }


    @Test
    fun `solve test`() {

        val procedurePrecision = 1e-5
        val rnd = Random(424242)

        // corner cases
        val a11 = mk.ndarray(mk[mk[4.0]])
        val b11 = mk.ndarray(mk[mk[3.0]])
        val b1 = mk.ndarray(mk[5.0])
        val b3 = mk.ndarray(mk[3.0, 4.0, 5.0])
        val b13 = mk.ndarray(mk[mk[3.0, 4.0, 5.0]])
        assertClose(solve(a11, b11), mk.ndarray(mk[mk[0.75]]), procedurePrecision)
        assertClose(solve(a11, b1), mk.ndarray(mk[1.25]), procedurePrecision)
        assertFailsWith<IllegalArgumentException>{solve(a11, b3)}
        assertClose(solve(a11, b13), mk.ndarray(mk[mk[0.75, 1.0, 1.25]]), procedurePrecision)

        for (iteration in 0 until 1000) {
            //test when second argument is d2 array
            val maxlen = 100
            val n = rnd.nextInt(1, maxlen)
            val m = rnd.nextInt(1, maxlen)
            val a = mk.d2array<Double>(n, n) { rnd.nextDouble() }
            val b = mk.d2array<Double>(n, m) { rnd.nextDouble() }
            assertClose(b, dot(a, solve(a, b)), procedurePrecision)


            //test when second argument is d1 vector
            val bd1 = mk.d1array(n) { rnd.nextDouble() }
            val sol = solve(a, bd1)
            assertClose(dot(a, sol.reshape(sol.shape[0], 1)).reshape(a.shape[0]), bd1, procedurePrecision)

        }
    }

    @Test
    fun `test upper hessenberg form`() {
        val n = 300
        val mat = getRandomMatrixComplexDouble(n, n)
        val (Q, H) = upperHessenberg(mat)

        // check H is upper Hessenberg
        for (i in 2 until H.shape[0]) {
            for (j in 0 until i - 1) {
                assertEquals(H[i, j], ComplexDouble.zero)
            }
        }

        // assert Q is unitary
        val approxId = tempDot(Q, Q.conjTranspose())
        val Id = mk.empty<ComplexDouble, D2>(n, n)

        assertCloseMatrixComplexDouble(approxId, idComplexDouble(n), 1e-5)

        // assert decomposition is valid
        val approxmat = tempDot(tempDot(Q, H), Q.conjTranspose())
        assertCloseMatrixComplexDouble(approxmat, mat, 1e-5);
    }


    @Test
    fun `test qr`() {
        val n = 100
        val mat = getRandomMatrixComplexDouble(n, n)
        val (q, r) = qrComplexDouble(mat)

        // assert decomposition is valid
        assertCloseMatrixComplexDouble(tempDot(q, r), mat, 1e-5)

        // assert q is unitary
        assertCloseMatrixComplexDouble(tempDot(q, q.conjTranspose()), idComplexDouble(n), 1e-5)

        // assert r is upper triangular
        for (i in 1 until r.shape[0]) {
            for (j in 0 until i) {
                if(r[i, j] != ComplexDouble.zero) {
                    assertEquals(r[i, j], ComplexDouble.zero)
                }
            }
        }
    }

    @Test
    fun `test Schur decomposition`() {
        for (attempt in 0 until 5) {
            val n = 100
            val mat = getRandomMatrixComplexDouble(n, n)
            val (q, r) = schurDecomposition(mat)

            // assert decomposition is valid
            assertCloseMatrixComplexDouble(tempDot(tempDot(q, r), q.conjTranspose()), mat, 1e-5)

            // assert q is unitary
            assertCloseMatrixComplexDouble(tempDot(q, q.conjTranspose()), idComplexDouble(n), 1e-5)

            // assert r is upper triangular
            for (i in 1 until r.shape[0]) {
                for (j in 0 until i) {
                    if (r[i, j] != ComplexDouble.zero) {
                        assertEquals(r[i, j], ComplexDouble.zero)
                    }
                }
            }
        }
    }


    @Test
    fun `test eigenvalues`() {
        val precision = 1e-2
        val n = 50
        val R = getRandomMatrixComplexDouble(n, n, -1000.0, 1000.0)
        for (i in 0 until R.shape[0]) {
            for (j in 0 until i) {
                R[i, j] = ComplexDouble.zero
            }
        }
        val Q = gramShmidtComplexDouble(getRandomMatrixComplexDouble(n, n))

        assertCloseMatrixComplexDouble(tempDot(Q, Q.conjTranspose()), idComplexDouble(n), precision = 1e-5)

        val mat = tempDot(tempDot(Q, R), Q.conjTranspose())

        var trueEigavals = List<ComplexDouble>(n) { i -> R[i, i] }

        val eigs = eigenvalues(mat)

        var testedEigenvals = List<ComplexDouble>(n) { i -> eigs[i] }

        trueEigavals = trueEigavals.sortedWith(compareBy({ it.re }, { it.im }))
        testedEigenvals = testedEigenvals.sortedWith(compareBy({ it.re }, { it.im }))



        for (i in 0 until n) {
            assertTrue("${trueEigavals[i]} =/= ${testedEigenvals[i]}") { (trueEigavals[i] - testedEigenvals[i]).abs() < precision}
        }

    }


    private fun assertCloseMatrixComplexDouble(a: MultiArray<ComplexDouble, D2>, b: MultiArray<ComplexDouble, D2>, precision: Double) {
        assertContentEquals(a.shape, b.shape, "matrices have different shapes")
        var maxabs = 0.0
        for (i in 0 until a.shape[0]) {
            for (j in 0 until a.shape[1]) {
                maxabs = max(maxabs, (a[i, j] - b[i, j]).abs())
            }
        }
        assertTrue(maxabs < precision, "matrices not close")

    }

    private fun <T : Number, D : Dim2> assertClose(a: MultiArray<T, D>, b: MultiArray<T, D>, precision: Double) {
        assertEquals(a.dim.d, b.dim.d, "matrices have different dimensions")
        assertContentEquals(a.shape, b.shape, "matrices have different shapes")
        var maxabs = 0.0
        if (a.dim.d == 1) {
            a as D1Array<T>
            b as D1Array<T>
            for (i in 0 until a.size) maxabs = max(abs(a[i].toDouble() - b[i].toDouble()), maxabs)
        } else {
            a as D2Array<T>
            b as D2Array<T>
            for (i in 0 until a.shape[0]) {
                for (j in 0 until a.shape[1]) {
                    val t = a[i, j].toDouble()
                    maxabs = max(abs(a[i, j].toDouble() - b[i, j].toDouble()), maxabs)
                }
            }
        }
        assertTrue(maxabs < precision, "matrices not close")
    }

    private fun <T : Number> assertTriangular(a: MultiArray<T, D2>, isLowerTriangular: Boolean, requireUnitsOnDiagonal: Boolean) {
        if (requireUnitsOnDiagonal) {
            for (i in 0 until min(a.shape[0], a.shape[1])) {
                if (a[i, i].toDouble() != 1.0)  throw AssertionError("element at position [$i, $i] of matrix \n$a\n is not unit")
            }
        }
        if (isLowerTriangular) {
            for (i in 0 until min(a.shape[0], a.shape[1])) {
                for (j in i + 1 until a.shape[1]) {
                    if(a[i, j].toDouble() != 0.0) throw AssertionError("element at position [$i, $j] of matrix \n$a\n is not zero")
                }
            }
        } else {
            for (i in 0 until min(a.shape[0], a.shape[1])) {
                for (j in 0 until i) {
                    if(a[i, j].toDouble() != 0.0) throw AssertionError("element at position [$i, $j] of matrix \n$a\n is not zero")
                }
            }
        }
    }

    fun getRandomMatrixComplexDouble(n: Int, m: Int, from: Double = 0.0, to: Double = 1.0, rnd: Random = Random(424242)): D2Array<ComplexDouble> {
        val a = mk.empty<ComplexDouble, D2>(n, m)

        for (i in 0 until n) {
            for (j in 0 until m) {
                a[i, j] = ComplexDouble(rnd.nextDouble()* (to - from) + from, rnd.nextDouble()* (to - from) + from)
            }
        }
        return a
    }

    fun idComplexDouble(n: Int): D2Array<ComplexDouble> {
        val ans = mk.empty<ComplexDouble, D2>(n, n)
        for (i in 0 until n) {
            ans[i, i] = 1.0.toComplexDouble()
        }
        return ans
    }

}