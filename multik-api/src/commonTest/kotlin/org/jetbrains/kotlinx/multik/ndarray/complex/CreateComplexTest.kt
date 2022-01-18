
class CreateComplexTest {

    @Test
    fun `test easy complex creation`() {
        assertEquals(Complex.i(0x01.toDouble()), 0x01.i)
        assertEquals(Complex.i(1.toShort().toDouble()), 1.toShort().i)
        assertEquals(Complex.i(1.toDouble()), 1.i)
        assertEquals(Complex.i(1L.toDouble()), 1L.i)
        assertEquals(Complex.i(1f), 1f.i)
        assertEquals(Complex.i(1.0), 1.0.i)

        assertEquals(ComplexFloat(1, 1), 1 + 1f.i)
        assertEquals(ComplexDouble(1, 1), 1 + 1.i)
    }
}