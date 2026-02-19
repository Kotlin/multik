# Solving systems of equations

<web-summary>
API reference for solving linear systems in Multik — solve() computes x in Ax = b
for a single right-hand side or multiple right-hand sides via mk.linalg.
</web-summary>

<card-summary>
solve() — solve Ax = b for single or multiple right-hand sides.
</card-summary>

<link-summary>
solve() — solving linear systems Ax = b.
</link-summary>

## Overview

The `solve` function solves a linear matrix equation **Ax = b**, where **A** is a square coefficient
matrix and **b** is either a single vector (1D) or a matrix of multiple right-hand sides (2D).

## Signatures

```kotlin
// Numeric — returns Double
fun <T : Number, D : Dim2> LinAlg.solve(
    a: MultiArray<T, D2>,
    b: MultiArray<T, D>
): NDArray<Double, D>

// Float — preserves Float precision
fun <D : Dim2> LinAlg.solve(
    a: MultiArray<Float, D2>,
    b: MultiArray<Float, D>
): NDArray<Float, D>

// Complex
fun <T : Complex, D : Dim2> LinAlg.solve(
    a: MultiArray<T, D2>,
    b: MultiArray<T, D>
): NDArray<T, D>
```

> The dimension parameter `D : Dim2` accepts both `D1` (vector) and `D2` (matrix),
> allowing `b` to be either a single right-hand side vector or a matrix of multiple vectors.
> {style="note"}

## Parameters

| Parameter | Type                | Description                                      |
|-----------|---------------------|--------------------------------------------------|
| `a`       | `MultiArray<T, D2>` | Square coefficient matrix (n x n).               |
| `b`       | `MultiArray<T, D>`  | Right-hand side: a vector (n) or matrix (n x m). |

**Returns:** `NDArray<Double, D>` (or `Float`/`Complex`) — the solution vector or matrix **x**.

## Example — single right-hand side

Solve the system:

- 2x + y = 5
- x + 3y = 7

```kotlin
val a = mk.ndarray(mk[mk[2.0, 1.0], mk[1.0, 3.0]])
val b = mk.ndarray(mk[5.0, 7.0])

val x = mk.linalg.solve(a, b)
// x ≈ [1.6, 1.8]

// Verify: a dot x ≈ b
```

## Example — multiple right-hand sides

Solve **Ax = B** where B has multiple columns:

```kotlin
val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])
val b = mk.ndarray(mk[mk[5.0, 6.0], mk[7.0, 8.0]])

val x = mk.linalg.solve(a, b)
// x: 2×2 matrix, each column is the solution for the corresponding column of b
```

## Pitfalls

* The coefficient matrix **A** must be **square** and **non-singular** (invertible). A singular matrix
  will cause an error.
* Numeric overloads return `Double` results, even if the inputs are `Int` or `Long`.
* The pure Kotlin engine supports `solve` for numeric types. Complex types require the OpenBLAS engine.

<seealso style="cards">
<category ref="api-docs">
  <a href="linear-algebra.md" summary="Overview of all linear algebra operations."/>
  <a href="matrix-operations.md" summary="inv, pow, QR, PLU, SVD decompositions."/>
  <a href="eigenvalues-and-eigenvectors.md" summary="Eigenvalue and eigenvector computation."/>
  <a href="matrix-norms.md" summary="Matrix and vector norms."/>
</category>
</seealso>