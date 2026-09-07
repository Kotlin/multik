# Multik

Kotlin Multiplatform ndarray library (math, linear algebra, statistics). Alpha, but the public API is
gated by the binary-compatibility-validator: **any public API change needs `./gradlew apiDump` and a
deliberate diff review.** Declarations marked `@ExperimentalMultikApi` are excluded from the dump —
mark new unstable APIs with it.

Base PRs against `develop`.

## Modules

| Module | Provides | Targets |
|---|---|---|
| `multik-core` | ndarray types, the `mk` entry point, `Math`/`LinAlg`/`Statistics` interfaces | all; CSV/NPY IO is JVM-only |
| `multik-kotlin` | `KEEngine`, pure Kotlin | all |
| `multik-openblas` | `NativeEngine`, OpenBLAS via C++/JNI (`multik_jni/`) | JVM + desktop Native only — no iOS/JS/WASM |
| `multik-default` | `DefaultEngine`: `NativeEngine` where available, else `KEEngine` | all |

Put new APIs in the broadest source set that supports them (`commonMain` first).

## Build and test

```bash
./gradlew assemble                     # needs gcc/g++/gfortran 8+ and JAVA_HOME
./gradlew assemble -x build_cmake      # skip the native OpenBLAS build
./gradlew :multik-core:jvmTest
./gradlew :multik-core:macosArm64Test  # or linuxX64Test / mingwX64Test
./gradlew apiCheck korroCheck          # run both before opening a PR
```

`:multik-openblas:jvmTest` needs the native library built by `build_cmake`; those tests extend
`NativeTestBase`, which loads it in `@BeforeTest`.

Name test functions `testXxx` — no backticks, unlike common Kotlin style.

## Public API

Follow the [Kotlin library guidelines](https://github.com/Kotlin/api-guidelines/tree/main/docs/topics).
Binary compatibility is the reason behind each of these:

- No data classes in public API — write an explicit `toString`.
- No new parameters on existing public functions, not even with defaults. Add overloads.
- Never widen or narrow an existing return type.
- Deprecate progressively: `@Deprecated` with `message` and `replaceWith`, WARNING → ERROR → HIDDEN.
- Compose with `NDArray<T, D>` / `MemoryView` / `Engine` instead of adding parallel hierarchies.
- Document view-vs-copy semantics on anything returning an `NDArray`.

## Errors

One condition, one exception type, identical in every engine — `multik-default` resolves the engine at
runtime, so a mismatch between `KEEngine` and `NativeEngine` makes user code platform-dependent.

- Numerical failure on valid input (singular matrix, no convergence) → `ArithmeticException`.
- No implementation for a dtype, dimension, format, or platform → `UnsupportedOperationException`.
- `EngineMultikException` is for engine discovery and loading only.
- LAPACK `info < 0` blames an argument *the wrapper* passed, so it is `check`/`error`, not `require`.
- Never `throw Exception`. Every message names the offending value — dtype, axis and its bound, shapes.
- Changing what an existing function throws is behaviour-visible: note it for the release notes.

## Docs

Writerside pages in `docs/` (`docs/mk.tree`); API docs via `./gradlew dokkaGenerate`.

Korro checks that the samples in `multik-core/src/commonTest/kotlin/samples/docs/` (delimited by
`// SampleStart` / `// SampleEnd`) match the code blocks in `docs/topics/**/*.md` — edit both together.

Skills: `multik-kdoc` for KDoc style and audits, `multik-docs` for Writerside pages.
