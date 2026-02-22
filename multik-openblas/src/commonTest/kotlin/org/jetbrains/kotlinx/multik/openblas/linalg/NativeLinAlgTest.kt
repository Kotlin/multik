package org.jetbrains.kotlinx.multik.openblas.linalg

import org.jetbrains.kotlinx.multik.api.ExperimentalMultikApi
import org.jetbrains.kotlinx.multik.api.arange
import org.jetbrains.kotlinx.multik.api.d1array
import org.jetbrains.kotlinx.multik.api.d2array
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.Norm
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.api.linalg.norm
import org.jetbrains.kotlinx.multik.api.linalg.solve
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble
import org.jetbrains.kotlinx.multik.ndarray.complex.ComplexFloat
import org.jetbrains.kotlinx.multik.ndarray.complex.i
import org.jetbrains.kotlinx.multik.ndarray.complex.plus
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.rangeTo
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.openblas.NativeTestBase
import org.jetbrains.kotlinx.multik.openblas.assertComplexFloatingNDArray
import org.jetbrains.kotlinx.multik.openblas.assertFloatingComplexNumber
import org.jetbrains.kotlinx.multik.openblas.assertFloatingNDArray
import org.jetbrains.kotlinx.multik.openblas.assertFloatingNumber
import kotlin.random.Random
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertEquals

class NativeLinAlgTest : NativeTestBase() {

    // --- Solve ---

    @Test
    fun `solve linear system F`() {
        // Expected values computed from Random(42) seed, 5x5 Float matrices
        val expected = mk.ndarray(
            mk[mk[4.1391945f, 1.2361444f, 4.4088345f],
                mk[-3.0071893f, 0.13484901f, -3.9121897f],
                mk[3.2885208f, -0.04077824f, 4.3054614f],
                mk[0.7955365f, 0.57545465f, 0.42709854f],
                mk[-11.024394f, -1.9956491f, -11.173507f]]
        )

        val (a, b) = data.getFloatMM(5, 5, 5, 3)

        assertFloatingNDArray(expected, NativeLinAlg.solve(a, b), epsilon = 1e5f)
    }

    @Test
    fun `solve linear system D`() {
        // Expected values computed from Random(42) seed, 5x5 Double matrices
        val expected = mk.ndarray(
            mk[mk[4.1391945, 1.2361444, 4.4088345],
                mk[-3.0071893, 0.13484901, -3.9121897],
                mk[3.2885208, -0.04077824, 4.3054614],
                mk[0.7955365, 0.57545465, 0.42709854],
                mk[-11.024394, -1.9956491, -11.173507]]
        )

        val (a, b) = data.getDoubleMM(5, 5, 5, 3)

        assertFloatingNDArray(expected, NativeLinAlg.solve(a, b), epsilon = 1e6)
    }

    @Test
    fun `solve linear system Complex`() {
        // Expected values computed from Random(42) seed, 3x3 ComplexDouble matrices
        val expected = mk.ndarray(
            mk[mk[ComplexDouble(-0.30547825, -0.61681154), ComplexDouble(0.41432816, -1.46046786),
                ComplexDouble(-0.35100211, 0.27240141)],
                mk[ComplexDouble(-0.14282777, 0.52435108), ComplexDouble(-0.14739684, 0.72480181),
                    ComplexDouble(0.75653133, -0.97962391)],
                mk[ComplexDouble(1.15623785, -0.11361717), ComplexDouble(0.65161407, 0.47386083),
                    ComplexDouble(0.51721532, 0.41166838)]]
        )

        val (a, b) = data.getComplexDoubleMM(3, 3, 3, 3)

        assertComplexFloatingNDArray(expected, NativeLinAlg.solve(a, b), epsilon = 1e8)
    }

    // --- Dot: vector ---

    @Test
    fun `simple vector dot test`() {
        val a = mk.ndarray(mk[1.0, 1.0])
        val b = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])

        assertEquals(3.0, b[0] dot a)
        assertEquals(7.0, b[1] dot a)
    }

    // --- Dot: matrix-matrix ---

    @Test
    fun `matrix-matrix dot test D`() {
        val expected = mk.ndarray(
            mk[mk[1.0853811780469889, 0.6321441231331913, 0.46677507285707914, 0.4892609866360924],
                mk[0.833116624067087, 0.7287671731075991, 0.7973174517659147, 0.8294695934205714],
                mk[0.5426264067593305, 0.4939259489941979, 0.5413707808847182, 0.5183069608507607],
                mk[1.048984456958365, 0.7710348889066437, 0.7189440755132327, 0.6763964597209662]]
        )

        val (matrix1, matrix2) = data.getDoubleMM(4, 3, 3, 4)

        val actual = NativeLinAlg.dot(matrix1, matrix2)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix-matrix dot test ComplexDouble`() {
        val expected = mk.ndarray(
            mk[mk[ComplexDouble(-11.0, 79.0), ComplexDouble(-7.0, 59.0), ComplexDouble(-3.0, +39.0)],
                mk[ComplexDouble(-9.0, 111.0), ComplexDouble(-5.0, 83.0), ComplexDouble(-1.0, 55.0)],
                mk[ComplexDouble(-7.0, 143.0), ComplexDouble(-3.0, 107.0), ComplexDouble(1.0, 71.0)]]
        )

        val matrix1 = mk.ndarray(
            mk[mk[ComplexDouble(1, 2), ComplexDouble(3, 4)],
                mk[ComplexDouble(2, 3), ComplexDouble(4, 5)],
                mk[ComplexDouble(3, 4), ComplexDouble(5, 6)]]
        )
        val matrix2 = mk.ndarray(
            mk[mk[ComplexDouble(9, 8), ComplexDouble(7, 6), ComplexDouble(5, 4)],
                mk[ComplexDouble(8, 7), ComplexDouble(6, 5), ComplexDouble(4, 3)]]
        )

        val actual = NativeLinAlg.dot(matrix1, matrix2)
        assertEquals(expected, actual)
    }

    @Test
    fun `matrix-matrix dot test ComplexFloat`() {
        val expected = mk.ndarray(
            mk[mk[ComplexFloat(-11.0, 79.0), ComplexFloat(-7.0, 59.0), ComplexFloat(-3.0, +39.0)],
                mk[ComplexFloat(-9.0, 111.0), ComplexFloat(-5.0, 83.0), ComplexFloat(-1.0, 55.0)],
                mk[ComplexFloat(-7.0, 143.0), ComplexFloat(-3.0, 107.0), ComplexFloat(1.0, 71.0)]]
        )

        val matrix1 = mk.ndarray(
            mk[mk[ComplexFloat(1, 2), ComplexFloat(3, 4)],
                mk[ComplexFloat(2, 3), ComplexFloat(4, 5)],
                mk[ComplexFloat(3, 4), ComplexFloat(5, 6)]]
        )
        val matrix2 = mk.ndarray(
            mk[mk[ComplexFloat(9, 8), ComplexFloat(7, 6), ComplexFloat(5, 4)],
                mk[ComplexFloat(8, 7), ComplexFloat(6, 5), ComplexFloat(4, 3)]]
        )

        val actual = NativeLinAlg.dot(matrix1, matrix2)
        assertComplexFloatingNDArray(expected, actual, 1e-4f)
    }

    @Test
    fun `matrix-matrix dot test F`() {
        val expected = mk.ndarray(
            mk[mk[0.8819745f, 0.64614516f, 0.7936589f, 0.5490592f],
                mk[0.543343f, 0.8133113f, 0.2793616f, 1.0130367f],
                mk[0.98215795f, 0.90664136f, 0.3652947f, 1.1545719f],
                mk[0.79763675f, 0.43727058f, 0.60035574f, 0.36558864f]]
        )

        val (matrix1, matrix2) = data.getFloatMM(4, 3, 3, 4)

        val actual = NativeLinAlg.dot(matrix1, matrix2)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix dot matrix transposed test`() {
        val (matrix1F, matrix2F) = data.getFloatMM(3, 4, 3, 4)
        val (matrix1D, matrix2D) = data.getDoubleMM(3, 4, 3, 4)

        val matrix1TF = matrix1F.transpose()
        val matrix1TFCopy = matrix1TF.deepCopy()
        val expectedF = NativeLinAlg.dot(matrix1TFCopy, matrix2F)
        val actualF = NativeLinAlg.dot(matrix1TF, matrix2F)
        assertFloatingNDArray(expectedF, actualF)


        val matrix1TD = matrix1D.transpose()
        val matrix1TDCopy = matrix1TD.deepCopy()
        val expectedD = NativeLinAlg.dot(matrix1TDCopy, matrix2D)
        val actualD = NativeLinAlg.dot(matrix1TD, matrix2D)
        assertFloatingNDArray(expectedD, actualD)

        val matrix2TF = matrix2F.transpose()
        val matrix2TFCopy = matrix2TF.deepCopy()
        val expected2F = NativeLinAlg.dot(matrix1F, matrix2TFCopy)
        val actual2F = NativeLinAlg.dot(matrix1F, matrix2TF)
        assertFloatingNDArray(expected2F, actual2F)

        val matrix2TD = matrix2D.transpose()
        val matrix2TDCopy = matrix2TD.deepCopy()
        val expected2D = NativeLinAlg.dot(matrix1D, matrix2TDCopy)
        val actual2D = NativeLinAlg.dot(matrix1D, matrix2TD)
        assertFloatingNDArray(expected2D, actual2D)
    }

    // --- Dot: matrix-vector ---

    @Test
    fun `matrix-vector dot test D`() {
        val expected = mk.ndarray(mk[0.8120680956454793, 0.676196362161166, 0.5898845530863276])

        val (matrix, vector) = data.getDoubleMV(3)

        val actual = NativeLinAlg.dot(matrix, vector)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix-vector dot test F`() {
        val expected = mk.ndarray(mk[0.86327714f, 0.3244831f, 0.76492393f])

        val (matrix, vector) = data.getFloatMV(3)

        val actual = NativeLinAlg.dot(matrix, vector)
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix slice dot vector test F`() {
        val (matrix, vector) = data.getFloatMV(5)
        val expected = NativeLinAlg.dot(matrix[2 until 5, 0 until 3].deepCopy(), vector[(0 until 5)..2].deepCopy())
        val actual = NativeLinAlg.dot(matrix[2 until 5, 0 until 3], vector[(0 until 5)..2])
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix slice dot vector test D`() {
        val (matrix, vector) = data.getDoubleMV(5)
        val expected = NativeLinAlg.dot(matrix[2 until 5, 0 until 3].deepCopy(), vector[(0 until 5)..2].deepCopy())
        val actual = NativeLinAlg.dot(matrix[2 until 5, 0 until 3], vector[(0 until 5)..2])
        assertFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix-vector dot test ComplexDouble`() {
        val matrix = mk.ndarray(
            mk[mk[ComplexDouble(1, 2), ComplexDouble(3, 4)],
                mk[ComplexDouble(5, 6), ComplexDouble(7, 8)]]
        )
        val vector = mk.ndarray(mk[ComplexDouble(1, 0), ComplexDouble(0, 1)])

        // [1+2i, 3+4i] . [1+0i, 0+1i] = (1+2i)*1 + (3+4i)*i = 1+2i + 3i-4 = -3+5i
        // [5+6i, 7+8i] . [1+0i, 0+1i] = (5+6i)*1 + (7+8i)*i = 5+6i + 7i-8 = -3+13i
        val expected = mk.ndarray(mk[ComplexDouble(-3, 5), ComplexDouble(-3, 13)])

        val actual = NativeLinAlg.dot(matrix, vector)
        assertComplexFloatingNDArray(expected, actual)
    }

    @Test
    fun `matrix-vector dot test ComplexFloat`() {
        val matrix = mk.ndarray(
            mk[mk[ComplexFloat(1, 2), ComplexFloat(3, 4)],
                mk[ComplexFloat(5, 6), ComplexFloat(7, 8)]]
        )
        val vector = mk.ndarray(mk[ComplexFloat(1, 0), ComplexFloat(0, 1)])

        val expected = mk.ndarray(mk[ComplexFloat(-3, 5), ComplexFloat(-3, 13)])

        val actual = NativeLinAlg.dot(matrix, vector)
        assertComplexFloatingNDArray(expected, actual, 1e-4f)
    }

    // --- Dot: vector-vector ---

    @Test
    fun `vector-vector dot test F`() {
        val (vector1, vector2) = data.getFloatVV(9)

        val actual = NativeLinAlg.dot(vector1, vector2)
        assertFloatingNumber(2.883776f, actual)
    }

    @Test
    fun `vector-vector dot test D`() {
        val (vector1, vector2) = data.getDoubleVV(9)

        val actual = NativeLinAlg.dot(vector1, vector2)
        assertFloatingNumber(1.9696041133566367, actual)
    }

    @Test
    fun `vector-vector dot test ComplexFloat`() {
        val random = Random(42)
        val vector1 = mk.d1array(9) { ComplexFloat(random.nextFloat(), random.nextFloat()) }
        val vector2 = mk.d1array(9) { ComplexFloat(random.nextFloat(), random.nextFloat()) }

        val actual = NativeLinAlg.dot(vector1, vector2)
        assertFloatingComplexNumber(0.23971967f + 4.6530027f.i, actual)
    }

    @Test
    fun `vector-vector dot test ComplexDouble from float`() {
        val random = Random(42)
        val vector1 = mk.d1array(9) { ComplexDouble(1.0, 0.0) }
        val vector2 = mk.d1array(9) { ComplexDouble(random.nextFloat(), random.nextFloat()) }

        val actual = NativeLinAlg.dot(vector1, vector2)
        assertFloatingComplexNumber(4.174294710159302 + 5.029674530029297.i, actual)
    }

    @Test
    fun `vector-vector dot test ComplexDouble`() {
        val random = Random(42)
        val vector1 = mk.d1array(9) { ComplexDouble(random.nextDouble(), random.nextDouble()) }
        val vector2 = mk.d1array(9) { ComplexDouble(random.nextDouble(), random.nextDouble()) }

        val actual = NativeLinAlg.dot(vector1, vector2)
        assertFloatingComplexNumber(0.17520206558121712 + 4.552420480555579.i, actual)
    }

    // --- Inverse ---

    @Test
    fun `compute inverse matrix of float`() {
        val a = data.getFloatM(117)
        val ainv = NativeLinAlg.inv(a)

        assertFloatingNDArray(mk.identity(117), NativeLinAlg.dot(a, ainv), 1e-4f)
    }

    @Test
    fun `compute inverse matrix of double`() {
        val a = data.getDoubleM(376)
        val ainv = NativeLinAlg.inv(a)

        assertFloatingNDArray(mk.identity(376), NativeLinAlg.dot(a, ainv))
    }

    @Test
    fun `compute inverse matrix of complex float`() {
        val a = data.getComplexFloatM(84)
        val ainv = NativeLinAlg.inv(a)

        assertComplexFloatingNDArray(mk.identity(84), NativeLinAlg.dot(a, ainv), 1e-4f)
    }

    @Test
    fun `compute inverse matrix of complex double`() {
        val a = data.getComplexDoubleM(83)
        val ainv = NativeLinAlg.inv(a)

        assertComplexFloatingNDArray(mk.identity(83), NativeLinAlg.dot(a, ainv))
    }

    // --- QR decomposition ---

    @Test
    fun `qr decomposition D`() {
        val a = data.getDoubleM(5, 4)
        val (q, r) = NativeLinAlgEx.qr(a)

        // Verify Q*R = A
        assertFloatingNDArray(a, NativeLinAlgEx.dotMM(q, r), epsilon = 1e-6)
        // Verify Q is orthogonal: Q^T * Q = I
        assertFloatingNDArray(mk.identity(4), NativeLinAlgEx.dotMM(q.transpose(), q), epsilon = 1e-6)
    }

    @Test
    fun `qr decomposition F`() {
        val a = data.getFloatM(5, 4)
        val (q, r) = NativeLinAlgEx.qrF(a)

        assertFloatingNDArray(a, NativeLinAlgEx.dotMM(q, r), epsilon = 1e-4f)
        assertFloatingNDArray(mk.identity(4), NativeLinAlgEx.dotMM(q.transpose(), q), epsilon = 1e-4f)
    }

    @Test
    fun `qr decomposition ComplexDouble`() {
        val a = data.getComplexDoubleM(4, 3)
        val (q, r) = NativeLinAlgEx.qrC(a)

        assertComplexFloatingNDArray(a, NativeLinAlgEx.dotMMComplex(q, r), epsilon = 1e-6)
    }

    // --- PLU decomposition ---

    @Test
    fun `plu decomposition D`() {
        val a = data.getDoubleM(5)
        val (p, l, u) = NativeLinAlgEx.plu(a)

        // Verify P * L * U = A
        val lu = NativeLinAlgEx.dotMM(l, u)
        val plu = NativeLinAlgEx.dotMM(p, lu)
        assertFloatingNDArray(a, plu, epsilon = 1e-8)
    }

    @Test
    fun `plu decomposition F`() {
        val a = data.getFloatM(5)
        val (p, l, u) = NativeLinAlgEx.pluF(a)

        val lu = NativeLinAlgEx.dotMM(l, u)
        val plu = NativeLinAlgEx.dotMM(p, lu)
        assertFloatingNDArray(a, plu, epsilon = 1e-5f)
    }

    @Test
    fun `plu decomposition ComplexDouble`() {
        val a = data.getComplexDoubleM(4)
        val (p, l, u) = NativeLinAlgEx.pluC(a)

        val lu = NativeLinAlgEx.dotMMComplex(l, u)
        val plu = NativeLinAlgEx.dotMMComplex(p, lu)
        assertComplexFloatingNDArray(a, plu, epsilon = 1e-8)
    }

    // --- SVD decomposition ---

    @Ignore // SVD requires quadmath
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun `svd decomposition D`() {
        val a = data.getDoubleM(4, 3)
        val (u, s, vt) = NativeLinAlgEx.svd(a)

        // Reconstruct: A = U[:, :k] * diag(S) * Vt[:k, :]
        val k = s.size
        val sigma = mk.zeros<Double>(k, k)
        for (i in 0 until k) sigma[i, i] = s[i]

        val uTrunc = u[0 until u.shape[0], 0 until k].deepCopy()
        val vtTrunc = vt[0 until k, 0 until vt.shape[1]].deepCopy()
        val reconstructed = NativeLinAlgEx.dotMM(NativeLinAlgEx.dotMM(uTrunc, sigma), vtTrunc)
        assertFloatingNDArray(a, reconstructed, epsilon = 1e-8)
    }

    @Ignore // SVD requires quadmath
    @OptIn(ExperimentalMultikApi::class)
    @Test
    fun `svd decomposition F`() {
        val a = data.getFloatM(4, 3)
        val (u, s, vt) = NativeLinAlgEx.svdF(a)

        val k = s.size
        val sigma = mk.zeros<Float>(k, k)
        for (i in 0 until k) sigma[i, i] = s[i]

        val uTrunc = u[0 until u.shape[0], 0 until k].deepCopy()
        val vtTrunc = vt[0 until k, 0 until vt.shape[1]].deepCopy()
        val reconstructed = NativeLinAlgEx.dotMM(NativeLinAlgEx.dotMM(uTrunc, sigma), vtTrunc)
        assertFloatingNDArray(a, reconstructed, epsilon = 1e-5f)
    }

    // --- Eigenvalues ---

    @Ignore // eig not implemented on JVM
    @Test
    fun `compute eigenvalues of diagonal matrix D`() {
        // Diagonal matrix has known eigenvalues equal to diagonal entries
        val a = mk.ndarray(
            mk[mk[3.0, 0.0, 0.0],
                mk[0.0, 1.0, 0.0],
                mk[0.0, 0.0, 2.0]]
        )
        val eigvals = NativeLinAlgEx.eigVals(a)

        // Sort eigenvalues by real part for deterministic comparison
        val sorted = (0 until eigvals.size).map { eigvals[it] }.sortedBy { it.re }
        assertFloatingComplexNumber(ComplexDouble(1.0, 0.0), sorted[0])
        assertFloatingComplexNumber(ComplexDouble(2.0, 0.0), sorted[1])
        assertFloatingComplexNumber(ComplexDouble(3.0, 0.0), sorted[2])
    }

    @Ignore // eig not implemented on JVM
    @Test
    fun `compute eigenvalues of diagonal matrix F`() {
        val a = mk.ndarray(
            mk[mk[3.0f, 0.0f],
                mk[0.0f, 5.0f]]
        )
        val eigvals = NativeLinAlgEx.eigValsF(a)

        val sorted = (0 until eigvals.size).map { eigvals[it] }.sortedBy { it.re.toDouble() }
        assertFloatingComplexNumber(ComplexFloat(3.0f, 0.0f), sorted[0], epsilon = 1e-5f)
        assertFloatingComplexNumber(ComplexFloat(5.0f, 0.0f), sorted[1], epsilon = 1e-5f)
    }

    @Ignore // eig not implemented on JVM
    @Test
    fun `compute eigenvalues and eigenvectors D`() {
        // Use symmetric matrix for reliable eigenvalue computation
        val a = mk.ndarray(mk[mk[2.0, 1.0], mk[1.0, 2.0]])
        val (w, v) = NativeLinAlgEx.eig(a)

        // Verify A * v_i = lambda_i * v_i for each eigenpair
        for (i in 0 until w.size) {
            val lambda = w[i]
            val eigvec = mk.d1array(a.shape[0]) { j -> v[j, i] }
            val av = mk.d1array(a.shape[0]) { j ->
                var sum = ComplexDouble.zero
                for (k in 0 until a.shape[1]) {
                    sum += ComplexDouble(a[j, k], 0.0) * eigvec[k]
                }
                sum
            }
            val lambdaV = mk.d1array(eigvec.size) { j -> lambda * eigvec[j] }
            assertComplexFloatingNDArray(lambdaV, av, epsilon = 1e-8)
        }
    }

    // --- Matrix pow ---

    @Test
    fun `matrix pow`() {
        val a = data.getDoubleM(4)

        // pow(A, 0) = I
        val powZero = NativeLinAlg.pow(a, 0)
        assertFloatingNDArray(mk.identity<Double>(4), powZero)

        // pow(A, 2) = A * A
        val powTwo = NativeLinAlg.pow(a, 2)
        val expected = NativeLinAlg.dot(a, a)
        assertFloatingNDArray(expected, powTwo, epsilon = 1e-8)

        // pow(A, 3) = A * A * A
        val powThree = NativeLinAlg.pow(a, 3)
        val expected3 = NativeLinAlg.dot(a, NativeLinAlg.dot(a, a))
        assertFloatingNDArray(expected3, powThree, epsilon = 1e-7)
    }

    // --- Norm ---

    @Test
    fun `compute norm`() {
        val a = mk.arange<Double>(9) - 4.0
        val b = a.reshape(3, 3)

        assertFloatingNumber(7.745966692414834, NativeLinAlgEx.norm(b, Norm.Fro))
        assertFloatingNumber(9.0, NativeLinAlgEx.norm(b, Norm.Inf))
        assertFloatingNumber(7.0, NativeLinAlgEx.norm(b, Norm.N1))
        assertFloatingNumber(4.0, NativeLinAlgEx.norm(b, Norm.Max))
    }

    @Test
    fun `compute norm F`() {
        val a = data.getFloatM(3)
        val aDouble = mk.d2array<Double>(3, 3) { a.data[it].toDouble() }

        // Float norm should be consistent with Double norm
        val normFro = NativeLinAlgEx.normF(a, Norm.Fro)
        val normFroD = NativeLinAlgEx.norm(aDouble, Norm.Fro)
        assertFloatingNumber(normFroD.toFloat(), normFro, epsilon = 1e-5f)

        val normInf = NativeLinAlgEx.normF(a, Norm.Inf)
        val normInfD = NativeLinAlgEx.norm(aDouble, Norm.Inf)
        assertFloatingNumber(normInfD.toFloat(), normInf, epsilon = 1e-5f)

        val normN1 = NativeLinAlgEx.normF(a, Norm.N1)
        val normN1D = NativeLinAlgEx.norm(aDouble, Norm.N1)
        assertFloatingNumber(normN1D.toFloat(), normN1, epsilon = 1e-5f)

        val normMax = NativeLinAlgEx.normF(a, Norm.Max)
        val normMaxD = NativeLinAlgEx.norm(aDouble, Norm.Max)
        assertFloatingNumber(normMaxD.toFloat(), normMax, epsilon = 1e-5f)
    }

    @Test
    fun `compute norm for vector`() {
        val vector = mk.ndarray(mk[1.1, 0.0, 3.2, 2.3, 5.0])

        // Vector norm goes through LinAlg extension function
        assertEquals(6.460650122085238, mk.linalg.norm(vector, Norm.Fro))
        assertEquals(11.600000000000001, mk.linalg.norm(vector, Norm.Inf))
        assertEquals(5.0, mk.linalg.norm(vector, Norm.N1))
        assertEquals(5.0, mk.linalg.norm(vector, Norm.Max))
    }
}
