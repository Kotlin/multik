# Matrix operations

<web-summary>
API reference for matrix operations in Multik — inverse, matrix power, QR decomposition,
PLU decomposition, and singular value decomposition (SVD).
</web-summary>

<card-summary>
Matrix inverse, power, QR, PLU, and SVD decompositions via mk.linalg.
</card-summary>

<link-summary>
Matrix operations — inv, pow, QR, PLU, SVD.
</link-summary>

## Matrix inverse

Computes the inverse of a square matrix.

### Signatures

{id="inv-signatures"}

```kotlin
fun <T : Number> LinAlg.inv(mat: MultiArray<T, D2>): NDArray<Double, D2>
fun LinAlg.inv(mat: MultiArray<Float, D2>): NDArray<Float, D2>
fun <T : Complex> LinAlg.inv(mat: MultiArray<T, D2>): NDArray<T, D2>
```

### Parameters

{id="inv-params"}

| Parameter | Type                | Description              |
|-----------|---------------------|--------------------------|
| `mat`     | `MultiArray<T, D2>` | Square matrix to invert. |

**Returns:** `NDArray` — the inverse matrix.

### Example

{id="inv-example"}

```kotlin
val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])

val aInv = mk.linalg.inv(a)
// [[-2.0,  1.0],
//  [ 1.5, -0.5]]

// Verify: a dot aInv ≈ identity
val identity = a dot aInv
// [[1.0, 0.0],
//  [0.0, 1.0]]
```

## Matrix power

Raises a square matrix to an integer power by repeated matrix multiplication.

### Signature

{id="pow-signature"}

```kotlin
fun <T : Number> LinAlg.pow(mat: MultiArray<T, D2>, n: Int): NDArray<T, D2>
```

### Parameters

{id="pow-params"}

| Parameter | Type                | Description                      |
|-----------|---------------------|----------------------------------|
| `mat`     | `MultiArray<T, D2>` | Square matrix.                   |
| `n`       | `Int`               | Exponent (non-negative integer). |

**Returns:** `NDArray<T, D2>` — the matrix raised to power `n`.

### Example

{id="pow-example"}

```kotlin
val a = mk.ndarray(mk[mk[1, 1], mk[1, 0]])

val a3 = mk.linalg.pow(a, 3)
// [[3, 2],
//  [2, 1]]
```

## QR decomposition

Decomposes a matrix into an orthogonal matrix **Q** and an upper-triangular matrix **R** such that
**A = QR**.

### Signatures

{id="qr-signatures"}

```kotlin
fun <T : Number> LinAlg.qr(mat: MultiArray<T, D2>): Pair<D2Array<Double>, D2Array<Double>>
fun LinAlg.qr(mat: MultiArray<Float, D2>): Pair<D2Array<Float>, D2Array<Float>>
fun <T : Complex> LinAlg.qr(mat: MultiArray<T, D2>): Pair<D2Array<T>, D2Array<T>>
```

### Parameters

{id="qr-params"}

| Parameter | Type                | Description   |
|-----------|---------------------|---------------|
| `mat`     | `MultiArray<T, D2>` | Input matrix. |

**Returns:** `Pair<Q, R>` where Q is orthogonal and R is upper-triangular.

### Example

{id="qr-example"}

```kotlin
val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])

val (q, r) = mk.linalg.qr(a)
// q: orthogonal matrix (Q^T * Q ≈ I)
// r: upper-triangular matrix
// a ≈ q dot r
```

## PLU decomposition

Decomposes a matrix into a permutation matrix **P**, a lower-triangular matrix **L**, and an
upper-triangular matrix **U** such that **A = P·L·U**.

### Signatures

{id="plu-signatures"}

```kotlin
fun <T : Number> LinAlg.plu(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D2Array<Double>, D2Array<Double>>
fun LinAlg.plu(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D2Array<Float>, D2Array<Float>>
fun <T : Complex> LinAlg.plu(mat: MultiArray<T, D2>): Triple<D2Array<T>, D2Array<T>, D2Array<T>>
```

### Parameters

{id="plu-params"}

| Parameter | Type                | Description   |
|-----------|---------------------|---------------|
| `mat`     | `MultiArray<T, D2>` | Input matrix. |

**Returns:** `Triple<P, L, U>` — permutation, lower-triangular, and upper-triangular matrices.

### Example

{id="plu-example"}

```kotlin
val a = mk.ndarray(mk[mk[2.0, 3.0], mk[5.0, 4.0]])

val (p, l, u) = mk.linalg.plu(a)
// p: permutation matrix
// l: lower-triangular (ones on diagonal)
// u: upper-triangular
// a ≈ p dot l dot u
```

## Singular value decomposition (SVD)

Decomposes a matrix into **U·S·V^T** where U and V are orthogonal matrices and S is a diagonal
matrix of singular values (returned as a 1D array).

> SVD is marked `@ExperimentalMultikApi` and may change in future releases.
> {style="warning"}

### Signatures

{id="svd-signatures"}

```kotlin
@ExperimentalMultikApi
fun <T : Number> LinAlg.svd(mat: MultiArray<T, D2>): Triple<D2Array<Double>, D1Array<Double>, D2Array<Double>>

@ExperimentalMultikApi
fun LinAlg.svd(mat: MultiArray<Float, D2>): Triple<D2Array<Float>, D1Array<Float>, D2Array<Float>>

@ExperimentalMultikApi
fun <T : Complex> LinAlg.svd(mat: MultiArray<T, D2>): Triple<D2Array<T>, D1Array<T>, D2Array<T>>
```

### Parameters

{id="svd-params"}

| Parameter | Type                | Description   |
|-----------|---------------------|---------------|
| `mat`     | `MultiArray<T, D2>` | Input matrix. |

**Returns:** `Triple<U, S, V>` where `S` is a 1D array of singular values.

### Example

{id="svd-example"}

```kotlin
@OptIn(ExperimentalMultikApi::class)
fun main() {
    val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])

    val (u, s, v) = mk.linalg.svd(a)
    // u: left singular vectors (2×2)
    // s: singular values [s1, s2]
    // v: right singular vectors (2×2)
}
```

## Pitfalls

* `inv` requires the matrix to be square and non-singular. A singular matrix will throw an exception.
* `pow` only accepts non-negative integer exponents.
* SVD is experimental (`@ExperimentalMultikApi`) — use `@OptIn(ExperimentalMultikApi::class)`.
* Decomposition functions (`qr`, `plu`, `svd`, `eig`) are not supported in the pure Kotlin engine
  for complex types — they require the OpenBLAS engine.

<seealso style="cards">
<category ref="api-docs">
  <a href="linear-algebra.md" summary="Overview of all linear algebra operations."/>
  <a href="dot.md" summary="Dot product operations."/>
  <a href="eigenvalues-and-eigenvectors.md" summary="Eigenvalue and eigenvector computation."/>
  <a href="solving-systems-of-equations.md" summary="Solving linear systems Ax = b."/>
  <a href="matrix-norms.md" summary="Matrix and vector norms."/>
</category>
</seealso>