# Random number generation

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for random number generation in Multik — mk.rand() creates NDArrays filled
with uniformly distributed random values.
</web-summary>

<card-summary>
Generate random NDArrays with mk.rand() — uniform distribution, ranges, seeds, and custom generators.
</card-summary>

<link-summary>
Random number generation — mk.rand() for uniformly distributed random arrays.
</link-summary>

## Overview

Multik provides the `mk.rand()` family of functions for creating arrays filled with random numbers.
All values follow a **uniform distribution**. You can control the range, seed, and random generator.

| Function                                | Description                                         |
|-----------------------------------------|-----------------------------------------------------|
| `mk.rand<T>(dim0, ...)`                 | Array with default random range for type `T`.       |
| `mk.rand<T, D>(from, until, dims)`      | Array with values in `[from, until)`.               |
| `mk.rand<T, D>(seed, from, until, ...)` | Reproducible random array with a seed.              |
| `mk.rand<T, D>(gen, from, until, ...)`  | Random array using a custom `kotlin.random.Random`. |

### Default ranges by type

| Type     | Range                              |
|----------|------------------------------------|
| `Int`    | `[Int.MIN_VALUE, Int.MAX_VALUE)`   |
| `Long`   | `[Long.MIN_VALUE, Long.MAX_VALUE)` |
| `Float`  | `[0.0, 1.0)`                       |
| `Double` | `[0.0, 1.0)`                       |

## Example

<!---FUN rand_example-->

```kotlin
val a = mk.rand<Double>(3, 3)                     // 3×3, values in [0.0, 1.0)
val b = mk.rand<Int, D1>(from = 1, until = 100, 10)  // 10 ints in [1, 100)
val c = mk.rand<Double, D2>(42, 0.0, 1.0, 3, 3)  // seeded 3×3
```

<!---END-->

## Pitfalls

* Complex types (`ComplexFloat`, `ComplexDouble`) are **not** supported. Use `d1array`/`d2array` with a custom init
  lambda for complex random arrays.
* The `from` bound is inclusive, the `until` bound is exclusive.

<seealso style="cards">
<category ref="api-docs">
  <a href="rand.md" summary="Full mk.rand() signature reference."/>
  <a href="array-creation.md" summary="All array creation functions."/>
</category>
</seealso>
