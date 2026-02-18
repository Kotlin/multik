# rand

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.rand() — create NDArrays filled with random values,
with optional range, seed, and custom Random generator.
</web-summary>

<card-summary>
Create arrays filled with random values. Supports ranges, seeds, and custom generators.
</card-summary>

<link-summary>
mk.rand() — random-filled arrays with optional range and seed.
</link-summary>

## Default random

```kotlin
inline fun <reified T : Number> Multik.rand(dim0: Int): D1Array<T>
inline fun <reified T : Number> Multik.rand(dim0: Int, dim1: Int): D2Array<T>
inline fun <reified T : Number> Multik.rand(dim0: Int, dim1: Int, dim2: Int): D3Array<T>
inline fun <reified T : Number> Multik.rand(
    dim0: Int, dim1: Int, dim2: Int, dim3: Int
): D4Array<T>
```

### From shape array

```kotlin
inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    shape: IntArray
): NDArray<T, D>
```

**Default distribution by type:**

| Type     | Range                              |
|----------|------------------------------------|
| `Int`    | `[Int.MIN_VALUE, Int.MAX_VALUE)`   |
| `Long`   | `[Long.MIN_VALUE, Long.MAX_VALUE)` |
| `Float`  | `[0.0, 1.0)`                       |
| `Double` | `[0.0, 1.0)`                       |

## Random in range

```kotlin
inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    from: T, until: T, vararg dims: Int
): NDArray<T, D>

inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    from: T, until: T, dims: IntArray
): NDArray<T, D>
```

Generates values uniformly distributed in `[from, until)`.

## Random with seed

```kotlin
inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    seed: Int, from: T, until: T, vararg dims: Int
): NDArray<T, D>

inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    seed: Int, from: T, until: T, dims: IntArray
): NDArray<T, D>
```

## Random with generator

```kotlin
inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    gen: Random, from: T, until: T, vararg dims: Int
): NDArray<T, D>

inline fun <reified T : Number, reified D : Dimension> Multik.rand(
    gen: Random, from: T, until: T, dims: IntArray
): NDArray<T, D>
```

## Parameters

| Parameter    | Type                       | Description                             |
|--------------|----------------------------|-----------------------------------------|
| `dim0..dim3` | `Int`                      | Axis sizes. Must be positive.           |
| `dims`       | `IntArray` or `vararg Int` | Shape of the output array.              |
| `from`       | `T`                        | Lower bound (inclusive).                |
| `until`      | `T`                        | Upper bound (exclusive).                |
| `seed`       | `Int`                      | Seed for the random generator.          |
| `gen`        | `Random`                   | Custom `kotlin.random.Random` instance. |

## Example

<!---FUN rand_example-->

```kotlin
val a = mk.rand<Double>(3, 3)                     // 3×3, values in [0.0, 1.0)
val b = mk.rand<Int, D1>(from = 1, until = 100, 10)  // 10 ints in [1, 100)
val c = mk.rand<Double, D2>(42, 0.0, 1.0, 3, 3)  // seeded 3×3
```

<!---END-->

> Complex types are **not** supported by `rand`. Use `d1array` or `d2array` with a custom init lambda for complex random
> arrays.
> {style="warning"}

<seealso style="cards">
<category ref="api-docs">
  <a href="zeros.md" summary="Create zero-filled arrays."/>
  <a href="ones.md" summary="Create one-filled arrays."/>
  <a href="d1array.md" summary="Create arrays with custom init for complex random values."/>
</category>
</seealso>
