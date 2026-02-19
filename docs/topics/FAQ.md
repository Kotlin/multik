# FAQ

<web-summary>
Frequently asked questions about Multik — setup, engines, platform support, common pitfalls,
and differences from NumPy.
</web-summary>

<card-summary>
Common questions about setup, engines, platform support, and usage.
</card-summary>

<link-summary>
FAQ — frequently asked questions about Multik.
</link-summary>

## Setup and installation

### I get "The map of engines is empty" or "Fail to find engine". What's wrong?

You need to add an **engine** dependency alongside `multik-core`. The core module only provides types and
interfaces — it does not include an implementation.

Add one of the following:

```kotlin
// Recommended — auto-selects the best engine per platform
implementation("org.jetbrains.kotlinx:multik-default:$multikVersion")

// Pure Kotlin only (all platforms, no native dependencies)
implementation("org.jetbrains.kotlinx:multik-kotlin:$multikVersion")
```

Make sure **all** Multik artifact versions match.

### The `multik-openblas` JAR is very large (~115 MB). Can I reduce it?

The JVM artifact bundles native libraries for all desktop platforms (Linux x64, macOS x64/arm64, Windows x64)
in a single JAR. Platform-specific artifacts are not yet available.

**Workaround:** If size is critical, use `multik-kotlin` instead — it is much smaller and has no native
dependencies.

### I get `GLIBC_2.35 not found` on Linux

The pre-built native library in `multik-openblas` requires glibc 2.35+. Older distributions
(e.g., Ubuntu 20.04) do not have it.

**Solutions:**

- Upgrade to Ubuntu 22.04+ or equivalent.
- Use `multik-kotlin` instead.
- Build the native library from source against your system's glibc.

## Engines

### How do I select or switch between engines?

`multik-default` auto-detects and loads the best engine. You can switch at runtime:

```kotlin
mk.setEngine(KEEngineType)     // Pure Kotlin engine
mk.setEngine(NativeEngineType) // OpenBLAS native engine
```

> Different engines may produce slightly different results for decompositions. For example,
> `linalg.qr` returns a reduced decomposition with the native engine but a full decomposition
> with the Kotlin engine.
> {style="note"}

### Why is the first Multik call slow?

The native engine (OpenBLAS) is lazily loaded on first use — the library is extracted from the JAR
and loaded via JNI. Subsequent calls are fast.

**Workarounds:**

- Perform a small warm-up call during application startup.
- Use `multik-kotlin` if startup latency matters more than computation speed.

## Platform support

### Does Multik work on Android?

Yes, but with caveats:

- **Use `multik-kotlin`**, not `multik-openblas` or `multik-default`. The OpenBLAS JAR bundles desktop
  `.so` files that cause build errors like `"libmultik_jni-linuxX64.so is not an ABI"` or AAB
  packaging failures.
- On older Android devices, `multik-default` may crash with `NoClassDefFoundError`.
- The native engine currently supports `arm64-v8a` only; `armv7` and `x86_64` Android ABIs are not
  supported.

### Does Multik work on Kotlin/Native, JS, and WASM?

`multik-kotlin` supports **JVM, JS, Native, and WASM** targets.

`multik-openblas` is available only on **JVM and desktop Native** (Linux x64, macOS x64/arm64, Windows x64).
It is **not** available on iOS, JS, or WASM.

> WASM support is experimental. Some operations (e.g., type casting in the multidimensional case) may
> throw exceptions.
> {style="warning"}

## Common pitfalls

### Reshaping a sliced array produces wrong data

Slicing returns a **view** that shares the underlying data buffer with non-contiguous strides.
`reshape` assumes contiguous memory, so it can produce incorrect results on views.

**Fix:** Call `.copy()` before reshaping:

```kotlin
val slice = array[0..2, 0..2]
val reshaped = slice.copy().reshape(1, 4)  // correct
```

### Mixing numeric types causes a runtime error

Multik does not support the generic `Number` type as an element type. When you mix types like `1.0`
(Double) and `2` (Int) in `mk[1.0, 2]`, the compiler infers `Number`, which compiles but **fails at
runtime**.

**Fix:** Ensure all elements use the same type:

```kotlin
mk.ndarray(mk[1.0, 2.0])   // all Double — correct
mk.ndarray(mk[1, 2])        // all Int — correct
// mk.ndarray(mk[1.0, 2])   // Number — runtime error!
```

### `dot` on sliced views can return incorrect results

Computing `dot` on a row extracted from a 2D array (which is a view) may produce wrong results,
especially with the native engine.

**Fix:** Copy the slice first:

```kotlin
val row = matrix[0].copy()   // explicit copy
val result = row dot vector   // now correct
```

### Integer division truncates

`mk.ndarray(mk[5]) / 2` yields `[2]`, not `[2.5]`. Use `Double` or `Float` arrays for
fractional results.

## NumPy comparison

### Does Multik support broadcasting?

**No.** Multik does not support NumPy-style broadcasting. Array-array operations require both operands
to have the same shape.

**Workaround:** Use `.map` or create a same-shaped array of the scalar value.

### Does Multik support boolean masks or `where`?

**No.** Element-wise comparisons that return boolean arrays (like `array == 0`), boolean indexing
(`array[mask]`), and `np.where(condition, x, y)` are not yet supported.

**Workaround:** Use `filter`, `mapIndexed`, or manual iteration:

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])
a.filter { it > 3 }   // [4, 5]
```

### How do I do matrix multiplication?

Use `mk.linalg.dot(a, b)` or the infix syntax `a dot b`. This handles matrix-matrix, matrix-vector,
and vector-vector products. See [dot](dot.md) for details.

> `*` is **element-wise** multiplication, not matrix multiplication.
> {style="warning"}

## I/O

### Can I read `.npy` files?

`.npy` format is supported on **JVM only**. It is not available on Native, JS, or WASM targets.

### Can I read CSV files with complex numbers?

CSV reading exists but has known issues when parsing complex numbers formatted by NumPy. As a
workaround, parse the file manually and construct the complex array in code.

<seealso style="cards">
<category ref="get-start">
  <a href="installation.md" summary="How to add Multik to your project."/>
  <a href="engines-of-multik.md" summary="Engines and how to choose between them."/>
  <a href="multik-on-different-platforms.md" summary="Platform-specific setup guides."/>
</category>
<category ref="ext">
  <a href="community-and-contribution.md" summary="Get help and contribute."/>
</category>
</seealso>