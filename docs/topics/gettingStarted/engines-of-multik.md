# Engines of Multik

<web-summary>
Understand how Multik separates its API from execution engines.
Learn the differences between DEFAULT, KOTLIN, and NATIVE engines and how to choose or set one.
</web-summary>

<card-summary>
Multik runs on pluggable engines.
Pick DEFAULT, KOTLIN, or NATIVE depending on platform and performance needs.
</card-summary>

<link-summary>
Learn how Multik engines work and how to select the right one.
</link-summary>

## Overview

Multik exposes a single API, but the actual math, linear algebra, and statistics work is performed by an **engine**.
Engines are interchangeable implementations, so the same ndarray code can run on different platforms and backends.

By default, Multik picks a suitable engine at runtime.
You can also select an engine explicitly when you need predictable performance or platform behavior.

## Built-in engines

| Engine      | Artifact          | Best for            | Notes                              |
|-------------|-------------------|---------------------|------------------------------------|
| **DEFAULT** | `multik-default`  | Most projects       | Chooses the best available engine. |
| **KOTLIN**  | `multik-kotlin`   | Maximum portability | Pure Kotlin, works everywhere.     |
| **NATIVE**  | `multik-openblas` | CPU-heavy math      | Uses OpenBLAS where supported.     |

## Selecting an engine

`Multik` exposes a map of available engines and lets you set the default one:

```kotlin
println(mk.engines.keys) // e.g., [DEFAULT, KOTLIN, NATIVE]
mk.setEngine(KEEngineType) // switch to the Kotlin engine
```

You can also check the current engine:

```kotlin
println(mk.engine)
```

If you try to select an engine that is not available, `mk.setEngine(...)` throws an `EngineMultikException`.

## Choosing the right engine

- **DEFAULT** - best for most users; a good balance of convenience and performance.
- **KOTLIN** - safest choice for multiplatform code or environments without native dependencies.
- **NATIVE** - best for CPU-heavy linear algebra when OpenBLAS is available.

For platform-specific details and support, see [](multik-on-different-platforms.md).

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="multik-on-different-platforms.md" summary="Overview of targets and engine availability."/>
  <a href="performance-and-optimization.md" summary="Performance tips: engines, copies, and allocation control."/>
  <a href="multik-on-JVM.md" summary="JVM-specific engine choices and setup notes."/>
</category>
<category ref="get-start">
  <a href="installation.md" summary="Add Multik dependencies to your project."/>
</category>
</seealso>
