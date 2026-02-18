# Dimension

<!---IMPORT samples.docs.apiDocs.ArrayObjects-->

<web-summary>
Reference for Multik's dimension type system — compile-time markers D1 through D4 and DN
that enforce array rank in the type signature.
</web-summary>

<card-summary>
Dimension markers D1–D4 and DN: compile-time rank safety for NDArrays.
</card-summary>

<link-summary>
Dimension types D1–D4 and DN — compile-time array rank markers.
</link-summary>

## Overview

Multik uses sealed dimension types as the second type parameter of `NDArray<T, D>`.
This gives compile-time safety for array rank: a `D2Array<Double>` cannot be accidentally passed where a
`D1Array<Double>` is expected.

## Dimension interface

```kotlin
public interface Dimension {
    public val d: Int
}
```

All dimension types implement `Dimension`. The `d` property holds the numeric rank.

## Hierarchy

```
Dimension
├── D1  (d = 1)  implements Dim1
├── D2  (d = 2)  implements Dim2
├── D3  (d = 3)  implements Dim3
├── D4  (d = 4)  implements Dim4
└── DN  (d = N)  implements DimN
```

Each concrete type extends a marker interface chain: `Dim1 ⊂ Dim2 ⊂ Dim3 ⊂ Dim4 ⊂ DimN`.
This means a function accepting `DimN` will also accept `D1`–`D4`.

## Concrete types

### D1

```kotlin
public sealed class D1(override val d: Int = 1) : Dimension, Dim1
```

Singleton companion: `D1`. Used for vectors.

### D2

```kotlin
public sealed class D2(override val d: Int = 2) : Dimension, Dim2
```

Singleton companion: `D2`. Used for matrices.

### D3

```kotlin
public sealed class D3(override val d: Int = 3) : Dimension, Dim3
```

Singleton companion: `D3`. Used for 3D tensors.

### D4

```kotlin
public sealed class D4(override val d: Int = 4) : Dimension, Dim4
```

Singleton companion: `D4`. Used for 4D tensors.

### DN

```kotlin
public class DN(override val d: Int) : Dimension, DimN
```

General N-dimensional type. Unlike `D1`–`D4`, `DN` is **not** sealed — the rank is stored in the `d` property and
checked at runtime.

## Type aliases

| Alias        | Meaning          |
|--------------|------------------|
| `D1Array<T>` | `NDArray<T, D1>` |
| `D2Array<T>` | `NDArray<T, D2>` |
| `D3Array<T>` | `NDArray<T, D3>` |
| `D4Array<T>` | `NDArray<T, D4>` |

## Helper functions

### dimensionOf

```kotlin
inline fun <reified D : Dimension> dimensionOf(dim: Int): D
```

Returns the dimension instance matching the given integer.
Throws `IllegalArgumentException` if `dim` does not match the reified type.

### dimensionClassOf

```kotlin
inline fun <reified D : Dimension> dimensionClassOf(dim: Int = -1): D
```

Resolves a dimension class by reified type parameter. When `dim` is provided, creates a `DN` with that rank.

## Usage

<!---FUN dim_example-->

```kotlin
val v: D1Array<Int> = mk.ndarray(mk[1, 2, 3])
v.dim   // D1
v.dim.d // 1

val m: D2Array<Double> = mk.zeros<Double>(3, 4)
m.dim   // D2
m.dim.d // 2
```

<!---END-->

### Dimension changes

Operations that change rank return a different dimension type:

<!---FUN change_dim_example-->

```kotlin
val m = mk.ndarray(mk[mk[1, 2], mk[3, 4]])                             // D2
val flat = m.flatten()                                                       // D1
val reshaped = m.reshape(1, 2, 2)                       // D3
val squeezed = m.reshape(1, 2, 2).squeeze(0)   // DN (d=2)
```

<!---END-->

> `squeeze`, `unsqueeze`, `view`, and `slice` return `NDArray<T, DN>` because the resulting rank depends on runtime
> arguments. Use `asD1Array()` … `asD4Array()` to restore a fixed dimension type when you know the rank.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="ndarray-class.md" summary="NDArray class — properties, operators, and methods."/>
  <a href="type.md" summary="DataType enum for runtime type identification."/>
</category>
</seealso>
