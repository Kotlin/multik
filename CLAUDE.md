# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multik is a Kotlin Multiplatform library for multidimensional arrays (ndarray), providing math, linear algebra, and
statistics operations. JetBrains incubator project (alpha stability). Group ID: `org.jetbrains.kotlinx`, version defined
in `gradle.properties`.

## Module Structure

- **multik-core** — Core ndarray types, the `Multik` (`mk`) entry-point object, and API interfaces (`Math`, `LinAlg`,
  `Statistics`). All platforms. JVM-only: CSV/NPY IO via `commons-csv` and `bio-npy`.
- **multik-kotlin** — Pure Kotlin engine (`KEEngine`). JVM + Native + JS + WASM.
- **multik-openblas** — Native engine (`NativeEngine`) using OpenBLAS via C++/JNI. JVM + desktop Native only (no
  iOS/JS/WASM). C++ source in `multik-openblas/multik_jni/`.
- **multik-default** — Combines multik-kotlin and multik-openblas via `DefaultEngine`/`DefaultEngineFactory`. Picks
  `NativeEngine` on supported platforms, falls back to `KEEngine`.

## Build Commands

```bash
# Build all modules (requires: JDK 8+, gcc/g++/gfortran 8+, JAVA_HOME set)
./gradlew assemble

# Build without native OpenBLAS (skip the CMake step)
./gradlew assemble -x build_cmake

# Build only core module
./gradlew :multik-core:build
```

CMake build uses env vars: `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`, `GCC_LIB_Path`, `TARGET_OS`.
Task chain: `createBuildDir` → `configureCmake` → `compileCmake` → `copyNativeLibs` → `build_cmake`.

## Testing

```bash
# JVM tests (JUnit Platform)
./gradlew :multik-core:jvmTest
./gradlew :multik-kotlin:jvmTest
./gradlew :multik-openblas:jvmTest   # requires built native lib (build_cmake)

# Single test class
./gradlew :multik-core:jvmTest --tests "org.jetbrains.kotlinx.multik.creation.Create2DIntArrayTests"

# Native tests (host target only)
./gradlew :multik-core:macosArm64Test   # or linuxX64Test, mingwX64Test
```

### Test Conventions

- Name test functions `testXxx` — no backticks in test names.
- Test classes use `*Test` / `*Tests` suffix.
- `multik-openblas` tests extend `NativeTestBase` which calls `libLoader("multik_jni").manualLoad()` in `@BeforeTest`.
  The JVM test task sets `java.library.path` to `build/cmake-build`.

## Architecture

The `Multik` object (aliased as `mk`) delegates to an `Engine` providing `Math`, `LinAlg`, and `Statistics`.
Engines: `KEEngine` (pure Kotlin, all platforms), `NativeEngine` (OpenBLAS, JVM + desktop Native),
`DefaultEngine` (picks best available). Engine loading is platform-specific via `expect fun enginesProvider()`.

## Code Conventions

- Kotlin 2.3.10 with `-Xexpect-actual-classes` compiler flag. JVM target: 1.8.
- All development happens on the `develop` branch — base PRs against `develop`.
- Convention plugins in `buildSrc/src/main/kotlin/`.

## API Design Guidelines

Follow
the [Kotlin library authors' guidelines](https://github.com/Kotlin/api-guidelines/tree/main/docs/topics). Key
Multik-specific rules:

- **No data classes in public API** — adding properties breaks binary compatibility. Use regular classes with explicit
  `toString`.
- **No new arguments on existing public functions**, even with defaults — breaks binary compatibility on JVM. Use manual
  overloads.
- **Never widen or narrow return types** of existing public functions.
- Use `@Deprecated` with `message`, `replaceWith`, and progressive `level` (WARNING → ERROR → HIDDEN).
- Use `@RequiresOptIn` (e.g. `@ExperimentalMultikApi`) for new unstable APIs.
- Core model is `NDArray<T, D>` + `MemoryView` + `Engine` — new operations should compose with these, not introduce
  parallel hierarchies.
- Place APIs in the broadest relevant source set (`commonMain` first).

## KDoc Guidelines

- Use `[ClassName]` / `[functionName]` references for cross-links.
- Document view-vs-copy semantics explicitly.
- See the `multik-kdoc` skill for the full KDoc style guide and audit workflow.

## Documentation

- Writerside docs in `docs/` (config: `docs/mk.tree`).
- API docs generated with Dokka: `./gradlew dokkaGenerate`.
- Doc samples validated by the Korro plugin. Samples live in `multik-core/src/commonTest/kotlin/samples/docs/` and map
  to `docs/topics/**/*.md`. Code blocks in sample files are delimited with `// SampleStart` and `// SampleEnd` comments.
  Keep samples and markdown in sync.
- See the `multik-docs` skill for Writerside page templates and conventions.
