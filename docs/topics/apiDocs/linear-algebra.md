# Linear algebra

<web-summary>
API reference for linear algebra operations in Multik — dot product, matrix inverse, matrix power,
decompositions (QR, PLU, SVD), eigenvalues, solving linear systems, and norms.
</web-summary>

<card-summary>
Linear algebra via mk.linalg — dot, inv, pow, decompositions, eigenvalues, solve, norms.
</card-summary>

<link-summary>
Linear algebra — dot, inv, pow, decompositions, eigenvalues, solve, norms.
</link-summary>

## Overview

The `mk.linalg` entry point provides linear algebra operations on 2D matrices and 1D vectors.
Functions are defined in the `LinAlg` and `LinAlgEx` interfaces; convenience extension functions
and infix operators are also available.

Most functions accept `Number`, `Float`, and `Complex` types through separate overloads.
Numeric overloads typically return `Double` results; float overloads preserve `Float` precision.

## Operations

| Operation                                                    | Function  | Description                                                           |
|--------------------------------------------------------------|-----------|-----------------------------------------------------------------------|
| [Dot product](dot.md)                                        | `dot`     | Matrix-matrix, matrix-vector, or vector-vector product.               |
| [Matrix inverse](matrix-operations.md#matrix-inverse)        | `inv`     | Inverse of a square matrix.                                           |
| [Matrix power](matrix-operations.md#matrix-power)            | `pow`     | Raise a square matrix to an integer power.                            |
| [QR decomposition](matrix-operations.md#qr-decomposition)    | `qr`      | Decompose into orthogonal Q and upper-triangular R.                   |
| [PLU decomposition](matrix-operations.md#plu-decomposition)  | `plu`     | Decompose into permutation P, lower-triangular L, upper-triangular U. |
| [SVD](matrix-operations.md#singular-value-decomposition-svd) | `svd`     | Singular value decomposition (experimental).                          |
| [Eigenvalues](eigenvalues-and-eigenvectors.md)               | `eig`     | Eigenvalues and eigenvectors.                                         |
| [Eigenvalues only](eigenvalues-and-eigenvectors.md#eigvals)  | `eigVals` | Eigenvalues without eigenvectors.                                     |
| [Solve](solving-systems-of-equations.md)                     | `solve`   | Solve a linear system Ax = b.                                         |
| [Norms](matrix-norms.md)                                     | `norm`    | Matrix or vector norm (Frobenius, 1-norm, infinity, max).             |

## Type support

Each operation has up to three overloads:

| Overload       | Input type                   | Output type                     |
|----------------|------------------------------|---------------------------------|
| Default        | `MultiArray<T: Number, D2>`  | `Double` / `NDArray<Double, …>` |
| Float (`…F`)   | `MultiArray<Float, D2>`      | `Float` / `NDArray<Float, …>`   |
| Complex (`…C`) | `MultiArray<T: Complex, D2>` | `NDArray<T, …>`                 |

## Quick example

```kotlin
val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
val b = mk.ndarray(mk[mk[5.0, 6.0], mk[7.0, 8.0]])

// Dot product
val c = mk.linalg.dot(a, b)        // 2×2 matrix product

// Inverse
val aInv = mk.linalg.inv(a)        // inverse of a

// Solve Ax = b
val x = mk.linalg.solve(a, mk.ndarray(mk[1.0, 2.0]))
```

<seealso style="cards">
<category ref="api-docs">
  <a href="dot.md" summary="Dot product — matrix-matrix, matrix-vector, vector-vector."/>
  <a href="matrix-operations.md" summary="inv, pow, QR, PLU, SVD decompositions."/>
  <a href="eigenvalues-and-eigenvectors.md" summary="Eigenvalue and eigenvector computation."/>
  <a href="solving-systems-of-equations.md" summary="Solving linear systems Ax = b."/>
  <a href="matrix-norms.md" summary="Matrix and vector norms."/>
  <a href="mathematical.md" summary="Mathematical functions via mk.math."/>
</category>
</seealso>