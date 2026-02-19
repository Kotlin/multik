# dot

<web-summary>
API reference for the dot product in Multik — matrix-matrix, matrix-vector,
and vector-vector multiplication via mk.linalg.dot() and the infix dot operator.
</web-summary>

<card-summary>
Dot product — matrix-matrix, matrix-vector, and vector-vector multiplication.
</card-summary>

<link-summary>
dot — matrix and vector multiplication.
</link-summary>

## Overview

The `dot` function computes the product of two arrays. The behavior depends on the operand dimensions:

| Left operand | Right operand | Result     | Operation                       |
|--------------|---------------|------------|---------------------------------|
| 2D matrix    | 2D matrix     | 2D matrix  | Standard matrix multiplication. |
| 2D matrix    | 1D vector     | 1D vector  | Matrix-vector multiplication.   |
| 1D vector    | 1D vector     | Scalar `T` | Inner (scalar) product.         |

## Signatures

### Via `mk.linalg`

```kotlin
// Matrix × Matrix
fun <T : Number> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>
fun <T : Complex> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D2>): NDArray<T, D2>

// Matrix × Vector
fun <T : Number> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>
fun <T : Complex> LinAlg.dot(a: MultiArray<T, D2>, b: MultiArray<T, D1>): NDArray<T, D1>

// Vector × Vector (scalar product)
fun <T : Number> LinAlg.dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
fun <T : Complex> LinAlg.dot(a: MultiArray<T, D1>, b: MultiArray<T, D1>): T
```

### Infix extension

```kotlin
infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2>
infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D2>): NDArray<T, D2>

infix fun <T : Number> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1>
infix fun <T : Complex> MultiArray<T, D2>.dot(b: MultiArray<T, D1>): NDArray<T, D1>

infix fun <T : Number> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T
infix fun <T : Complex> MultiArray<T, D1>.dot(b: MultiArray<T, D1>): T
```

## Parameters

| Parameter | Type                                       | Description    |
|-----------|--------------------------------------------|----------------|
| `a`       | `MultiArray<T, D2>` or `MultiArray<T, D1>` | Left operand.  |
| `b`       | `MultiArray<T, D2>` or `MultiArray<T, D1>` | Right operand. |

**Returns:** `NDArray<T, D2>`, `NDArray<T, D1>`, or scalar `T` depending on operand dimensions.

## Matrix × Matrix

```kotlin
val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
val b = mk.ndarray(mk[mk[5, 6], mk[7, 8]])

val c = mk.linalg.dot(a, b)
// [[19, 22],
//  [43, 50]]

// Or using infix syntax:
val d = a dot b
```

## Matrix × Vector

```kotlin
val mat = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
val vec = mk.ndarray(mk[10.0, 20.0])

val result = mk.linalg.dot(mat, vec)   // [50.0, 110.0]

// Infix:
val result2 = mat dot vec               // [50.0, 110.0]
```

## Vector × Vector (scalar product)

```kotlin
val u = mk.ndarray(mk[1.0, 2.0, 3.0])
val v = mk.ndarray(mk[4.0, 5.0, 6.0])

val scalar = mk.linalg.dot(u, v)   // 32.0  (1*4 + 2*5 + 3*6)

// Infix:
val scalar2 = u dot v               // 32.0
```

## Pitfalls

* This is **matrix multiplication**, not element-wise multiplication. For element-wise `*`, see
  [Arithmetic Operations](arithmetic-operations.md).
* For matrix-matrix: the number of columns in `a` must equal the number of rows in `b`.
* For matrix-vector: the vector length must equal the number of columns in the matrix.
* For vector-vector: both vectors must have the same length.

<seealso style="cards">
<category ref="api-docs">
  <a href="linear-algebra.md" summary="Overview of all linear algebra operations."/>
  <a href="matrix-operations.md" summary="inv, pow, QR, PLU, SVD decompositions."/>
  <a href="arithmetic-operations.md" summary="Element-wise arithmetic operations."/>
</category>
</seealso>
