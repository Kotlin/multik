# Matrix norms

<web-summary>
API reference for matrix and vector norms in Multik — Frobenius, 1-norm, infinity norm, and max norm
via mk.linalg.norm().
</web-summary>

<card-summary>
Matrix and vector norms — Frobenius, 1-norm, infinity, and max.
</card-summary>

<link-summary>
norm() — matrix and vector norms.
</link-summary>

## Overview

The `norm` function computes the norm of a matrix or vector. The norm type is selected via the `Norm` enum.

## Norm types

| Enum value | Formula                          | Description                                              |
|------------|----------------------------------|----------------------------------------------------------|
| `Norm.Fro` | sqrt(sum of \|a_ij\|^2)          | Frobenius norm (square root of sum of squares). Default. |
| `Norm.N1`  | max over columns of sum \|a_ij\| | Maximum column sum (1-norm).                             |
| `Norm.Inf` | max over rows of sum \|a_ij\|    | Maximum row sum (infinity norm).                         |
| `Norm.Max` | max(\|a_ij\|)                    | Maximum absolute element value.                          |

## Signatures

```kotlin
// Double matrix
fun LinAlg.norm(mat: MultiArray<Double, D2>, norm: Norm = Norm.Fro): Double

// Float matrix
fun LinAlg.norm(mat: MultiArray<Float, D2>, norm: Norm = Norm.Fro): Float

// Double vector
fun LinAlg.norm(mat: MultiArray<Double, D1>, norm: Norm = Norm.Fro): Double

// Float vector
fun LinAlg.norm(mat: MultiArray<Float, D1>, norm: Norm = Norm.Fro): Float
```

## Parameters

| Parameter | Type                                                             | Description                     |
|-----------|------------------------------------------------------------------|---------------------------------|
| `mat`     | `MultiArray<Double/Float, D2>` or `MultiArray<Double/Float, D1>` | Input matrix or vector.         |
| `norm`    | `Norm`                                                           | Norm type. Default: `Norm.Fro`. |

**Returns:** `Double` or `Float` — the computed norm value.

## Example

```kotlin
val a = mk.ndarray(mk[mk[1.0, 2.0], mk[3.0, 4.0]])

mk.linalg.norm(a)                // Frobenius: sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
mk.linalg.norm(a, Norm.N1)       // max column sum: max(4, 6) = 6.0
mk.linalg.norm(a, Norm.Inf)      // max row sum: max(3, 7) = 7.0
mk.linalg.norm(a, Norm.Max)      // max |element|: 4.0
```

### Vector norm

```kotlin
val v = mk.ndarray(mk[3.0, 4.0])

mk.linalg.norm(v)                // Frobenius: sqrt(9+16) = 5.0
```

> `norm` only accepts `Double` and `Float` arrays. For integer arrays, convert with `toType<Double>()`
> first.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="linear-algebra.md" summary="Overview of all linear algebra operations."/>
  <a href="matrix-operations.md" summary="inv, pow, QR, PLU, SVD decompositions."/>
  <a href="dot.md" summary="Dot product operations."/>
</category>
</seealso>