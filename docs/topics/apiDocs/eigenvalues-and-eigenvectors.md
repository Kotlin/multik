# Eigenvalues and eigenvectors

<web-summary>
API reference for eigenvalue and eigenvector computation in Multik — eig() for values and vectors,
eigVals() for values only, via mk.linalg.
</web-summary>

<card-summary>
Eigenvalue decomposition — eig() and eigVals() via mk.linalg.
</card-summary>

<link-summary>
Eigenvalues and eigenvectors — eig() and eigVals().
</link-summary>

## Overview

Multik provides two functions for eigenvalue computation:

| Function  | Description                                                             |
|-----------|-------------------------------------------------------------------------|
| `eig`     | Computes both eigenvalues and eigenvectors.                             |
| `eigVals` | Computes eigenvalues only (more efficient when vectors are not needed). |

Both functions return complex results because eigenvalues of a real matrix can be complex.

## eig

Computes the eigenvalues and eigenvectors of a square matrix.

### Signatures

{id="eig-signatures"}

```kotlin
fun <T : Number> LinAlg.eig(mat: MultiArray<T, D2>): Pair<D1Array<ComplexDouble>, D2Array<ComplexDouble>>
fun LinAlg.eig(mat: MultiArray<Float, D2>): Pair<D1Array<ComplexFloat>, D2Array<ComplexFloat>>
fun <T : Complex> LinAlg.eig(mat: MultiArray<T, D2>): Pair<D1Array<T>, D2Array<T>>
```

### Parameters

{id="eig-params"}

| Parameter | Type                | Description                 |
|-----------|---------------------|-----------------------------|
| `mat`     | `MultiArray<T, D2>` | Square matrix to decompose. |

**Returns:** `Pair<eigenvalues, eigenvectors>` where:

- `eigenvalues` — a 1D array of complex eigenvalues.
- `eigenvectors` — a 2D matrix whose columns are the corresponding eigenvectors.

### Example

{id="eig-example"}

```kotlin
val a = mk.ndarray(mk[mk[4.0, -2.0], mk[1.0, 1.0]])

val (values, vectors) = mk.linalg.eig(a)
// values: [3.0+0.0i, 2.0+0.0i]
// vectors: columns are eigenvectors for each eigenvalue
```

## eigVals

{id="eigvals"}

Computes only the eigenvalues of a square matrix (without eigenvectors).

### Signatures

{id="eigvals-signatures"}

```kotlin
fun <T : Number> LinAlg.eigVals(mat: MultiArray<T, D2>): D1Array<ComplexDouble>
fun LinAlg.eigVals(mat: MultiArray<Float, D2>): D1Array<ComplexFloat>
fun <T : Complex> LinAlg.eigVals(mat: MultiArray<T, D2>): D1Array<T>
```

### Parameters

{id="eigvals-params"}

| Parameter | Type                | Description    |
|-----------|---------------------|----------------|
| `mat`     | `MultiArray<T, D2>` | Square matrix. |

**Returns:** `D1Array` of complex eigenvalues.

### Example

{id="eigvals-example"}

```kotlin
val a = mk.ndarray(mk[mk[2.0, 1.0], mk[1.0, 2.0]])

val eigenvalues = mk.linalg.eigVals(a)
// [3.0+0.0i, 1.0+0.0i]
```

## Pitfalls

* Eigenvalue results are always **complex** (`ComplexDouble` or `ComplexFloat`), even for symmetric matrices
  where all eigenvalues are real. Access the real part with `.re` and imaginary part with `.im`.
* The input matrix must be **square**. Non-square matrices will cause an error.
* The pure Kotlin engine does **not** support `eig` or `eigVals` — use the OpenBLAS or default engine.

<seealso style="cards">
<category ref="api-docs">
  <a href="linear-algebra.md" summary="Overview of all linear algebra operations."/>
  <a href="matrix-operations.md" summary="inv, pow, QR, PLU, SVD decompositions."/>
  <a href="solving-systems-of-equations.md" summary="Solving linear systems Ax = b."/>
</category>
</seealso>