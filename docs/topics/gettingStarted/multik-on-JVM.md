# Multik on JVM

<web-summary>
Use Multik on the JVM with either the pure Kotlin engine or the OpenBLAS-backed native engine.
Learn how to pick the right dependency and switch engines when needed.
</web-summary>

<card-summary>
JVM is the most flexible Multik target.
Choose KOTLIN for portability or NATIVE for faster linear algebra.
</card-summary>

<link-summary>
Run Multik on the JVM with KOTLIN or NATIVE engines and select the best fit for your workload.
</link-summary>

## Overview

The JVM is the most flexible Multik target.
You can use a pure Kotlin engine for portability or a native OpenBLAS engine for performance.

## Engine availability

- **DEFAULT** (`multik-default`) - picks the best available engine.
- **KOTLIN** (`multik-kotlin`) - pure Kotlin implementation.
- **NATIVE** (`multik-openblas`) - OpenBLAS-backed engine.

## Dependencies

Pick one of these options:

- `multik-default` for convenience.
- `multik-kotlin` if you want pure Kotlin only.
- `multik-openblas` if you want the native engine explicitly.

See [](installation.md) for Gradle snippets.

> You can switch engines at runtime:
{style="tip"}

```kotlin
mk.setEngine(KEEngineType)        // pure Kotlin
// mk.setEngine(NativeEngineType)  // OpenBLAS, if available
```

Check the current engine:

```kotlin
println(mk.engine)
```

Other considerations:

- OpenBLAS is useful for heavy linear algebra.
- Native dependencies must be available and compatible with your environment.
- On Android, the OpenBLAS JVM engine supports arm64-v8a only.

For platform-level constraints, see [](multik-on-different-platforms.md).

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="engines-of-multik.md" summary="Understand DEFAULT, KOTLIN, and NATIVE engines."/>
  <a href="multik-on-desktop.md" summary="Desktop JVM/Native guidance and OpenBLAS notes."/>
  <a href="multik-on-different-platforms.md" summary="Overview of Multik targets and engine availability."/>
</category>
<category ref="get-start">
  <a href="installation.md" summary="Add Multik dependencies to your project."/>
  <a href="multik-on-different-platforms.md" summary="Full list of Kotlin target presets supported by Multik."/>
</category>
</seealso>
