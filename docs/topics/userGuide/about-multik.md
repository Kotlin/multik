# About Multik

<!---IMPORT samples.docs.userGuide.AboutMultik-->

<web-summary>
Get a concise introduction to Multik: what it is, what problems it solves, and how its core concepts fit together.
Learn about ndarrays, engines, and the modules you combine to build fast, type-safe numerical code in Kotlin.
</web-summary>

<card-summary>
Meet Multik, a multiplatform ndarray library for Kotlin.
Understand its goals, core concepts, and how its engines and modules work together.
</card-summary>

<link-summary>
Learn what Multik is, why it exists, and how its ndarrays and engines fit into Kotlin projects.
</link-summary>

## Overview

Multik is a Kotlin Multiplatform library for working with numerical data in multidimensional arrays.
It provides a compact, type-safe API for math, linear algebra, and statistics,
while keeping the code Kotlin-first and readable.

Multik is a good fit when Kotlin collections become too verbose or too slow for numeric work,
such as data preparation, feature engineering, algorithm prototyping, or scientific computing.

## Why use Multik?

- **NDArray as a first-class type** for dense, homogeneous data.
- **Static typing of dtype and dimension** to catch mistakes early.
- **Performance-oriented storage** built on contiguous primitive arrays.
- **Multiple execution engines** so you can choose between portability and native speed.
- **Multiplatform support** across JVM, JS, Native, and WASM.

## Core concepts

The central type in Multik is `NDArray`, a container for numeric data with a fixed element type and dimension.
Key terms:

- **Dimension** - number of axes (1D, 2D, 3D, or ND).
- **Shape** - size of each axis.
- **Strides** - steps in storage needed to move along each axis.
- **dtype** - the element type, such as `Int`, `Double`, or complex numbers.
- **Engine** - the implementation that executes math, linear algebra, and statistics.

## Modules and engines

Multik separates the API from the execution engine:

- `multik-core` defines ndarray types and the API surface.
- `multik-kotlin` provides a pure Kotlin engine.
- `multik-openblas` provides a native engine backed by OpenBLAS.
- `multik-default` bundles the Kotlin and OpenBLAS engines.

This lets you pick the best engine for your platform and performance needs without changing your code.

## A tiny example

<!---FUN tiny_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
val b = mk.ndarray(mk[4, 5, 6])
val c = a + b
println(c) // [5, 7, 9]
```

<!---END-->

Even simple operations become concise and expressive when you use ndarrays.

<seealso style="cards" title="Next steps">
<category ref="get-start">
  <a href="quickstart.md" summary="Build your first ndarray workflows in minutes."/>
  <a href="installation.md" summary="Add Multik dependencies to your project."/>
</category>
<category ref="user-guide">
  <a href="basic.md" summary="Learn the NDArray model and core operations.">Basic</a>
  <a href="creating-multidimensional-arrays.md" summary="Create ndarrays from Kotlin collections and primitives."/>
  <a href="engines-of-multik.md" summary="Understand DEFAULT, KOTLIN, and NATIVE engines."/>
  <a href="multik-on-different-platforms.md" summary="See how Multik works across JVM, JS, Native, and WASM."/>
</category>
</seealso>
