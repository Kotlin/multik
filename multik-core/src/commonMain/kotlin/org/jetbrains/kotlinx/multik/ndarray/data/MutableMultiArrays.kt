package org.jetbrains.kotlinx.multik.ndarray.data

/**
 * Mutable extension of [MultiArray] that allows element modification.
 *
 * Exposes a writable [data] buffer and narrows return types of structural operations
 * ([copy], [deepCopy], [reshape], [transpose], [squeeze], [unsqueeze]) to [MutableMultiArray]
 * so that the result remains writable.
 *
 * @param T the element type.
 * @param D the dimension type.
 * @see [MultiArray]
 * @see [NDArray]
 */
public interface MutableMultiArray<T, D : Dimension> : MultiArray<T, D> {
    public override val data: MemoryView<T>

    override fun copy(): MutableMultiArray<T, D>

    override fun deepCopy(): MutableMultiArray<T, D>

    // Reshape

    override fun reshape(dim1: Int): MutableMultiArray<T, D1>

    override fun reshape(dim1: Int, dim2: Int): MutableMultiArray<T, D2>

    override fun reshape(dim1: Int, dim2: Int, dim3: Int): MutableMultiArray<T, D3>

    override fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int): MutableMultiArray<T, D4>

    override fun reshape(dim1: Int, dim2: Int, dim3: Int, dim4: Int, vararg dims: Int): MutableMultiArray<T, DN>

    override fun transpose(vararg axes: Int): MutableMultiArray<T, D>

    override fun squeeze(vararg axes: Int): MutableMultiArray<T, DN>

    override fun unsqueeze(vararg axes: Int): MutableMultiArray<T, DN>
}
