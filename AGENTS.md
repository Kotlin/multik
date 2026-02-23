# Repository Guidelines

## Project Overview
Multik is a Kotlin Multiplatform library for multidimensional arrays (ndarray) with math, linear algebra, and statistics APIs.

## Project Structure & Modules
- `multik-core/` contains core ndarray types, the `Multik` (`mk`) entry point, and API interfaces (`Math`, `LinAlg`, `Statistics`).
- `multik-kotlin/` provides the pure Kotlin engine (`KEEngine`) for JVM + Native + JS + WASM.
- `multik-openblas/` provides the OpenBLAS native engine (`NativeEngine`) for JVM + desktop Native. Native C++/JNI lives in `multik-openblas/multik_jni/`.
- `multik-default/` wires engines via `DefaultEngine`/`DefaultEngineFactory`.
- Tests: `multik-core/src/commonTest` (common), `multik-core/src/jvmTest` (JVM).
- Doc samples: `multik-core/src/commonTest/kotlin/samples/docs/`, validated against `docs/topics/`.
- Build logic: `buildSrc/`, Gradle config in `build.gradle.kts`, `settings.gradle.kts`, `gradle/`.

## Build & Test
- Build all modules: `./gradlew assemble` (requires JDK 8+, `JAVA_HOME`, and `gcc`/`g++`/`gfortran` 8+).
- Skip native OpenBLAS: `./gradlew assemble -x build_cmake`.
- Build core only: `./gradlew :multik-core:build`.
- JVM tests: `./gradlew :multik-core:jvmTest`, `./gradlew :multik-kotlin:jvmTest`, `./gradlew :multik-openblas:jvmTest`.
- Single JVM test class: `./gradlew :multik-core:jvmTest --tests "org.jetbrains.kotlinx.multik.SomeTest"`.
- Native tests (host only): `./gradlew :multik-core:macosArm64Test` (or `linuxX64Test`, `mingwX64Test`).
- JS/WASM tests are disabled in build config (`testTask { enabled = false }`).

## Architecture Notes
- `Multik` (`mk`) delegates to an `Engine` which provides `Math`, `LinAlg`, and `Statistics`.
- Engine selection is platform-specific via `enginesProvider()` (`ServiceLoader` on JVM, eager init on Native/JS/WASM).
- Array model: `MultiArray<T, D>` (read-only) → `MutableMultiArray<T, D>` → `NDArray<T, D>` (concrete).
- Views vs copies: slicing/transpose/reshape/squeeze return views; `copy()` duplicates buffer; `deepCopy()` makes contiguous data.

## API Design & Library Author Guidelines (Apply to Public API)
- Minimize mental complexity: keep the API small, composable, and predictable; build on core abstractions (`NDArray`, `Engine`).
- Readability: prefer extension functions for non-core behavior, avoid boolean parameters in public APIs, use domain terms.
- Consistency: keep parameter order/naming aligned across the API; use `OrNull`/`Catching` suffixes consistently.
- Predictability: sensible defaults, sealed hierarchies for closed domains, avoid exposing mutable state; validate with `require`/`check`.
- Debuggability: implement meaningful, consistent `toString()` on stateful types.
- Testability: avoid global mutable state; allow injection or configuration (e.g., `mk.setEngine()`).

## Backward Compatibility
- Avoid data classes in public API; they break binary compatibility when fields change.
- Do not add parameters to existing public functions (even with defaults). Prefer manual overloads.
- Do not widen or narrow return types of public functions.
- Use a deprecation cycle with `@Deprecated` (`WARNING` → `ERROR` → `HIDDEN`) and `replaceWith`.
- Use `@RequiresOptIn` for experimental APIs and document guarantees.
- Consider using the Binary Compatibility Validator.

## Documentation
- KDoc all public APIs. Start with what the API does, document inputs, outputs, exceptions, and edge cases.
- For lambdas, document exception behavior and threading/ordering guarantees.
- Keep `docs/` updated for public API or behavior changes.
- Doc samples in `multik-core/src/commonTest/kotlin/samples/docs/` are validated by Korro against `docs/topics/`.
- Generate API docs with `./gradlew dokkaGenerate`.

## Multiplatform Guidance
- Place APIs in the broadest source set possible: `commonMain` → intermediate source sets → platform-specific.
- Ensure consistent behavior across platforms and document unavoidable differences.
- Use `expect`/`actual` for platform-specific implementations with identical contracts.
- OpenBLAS is JVM + desktop Native only; JS/WASM/iOS use the Kotlin engine.
