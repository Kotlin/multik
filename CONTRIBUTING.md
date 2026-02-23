# Contributing Guidelines

There are two main ways to contribute to the project &mdash; submitting issues and submitting
fixes/changes/improvements via pull requests.

## Submitting issues

Both bug reports and feature requests are welcome.
Submit issues [here](https://github.com/Kotlin/multik/issues).

* Search for existing issues to avoid reporting duplicates.
* When submitting a bug report:
    * Use a 'bug report' template when creating a new issue.
    * Test it against the most recently released version. It might have been already fixed.
    * Include the Multik version, module (`multik-default`, `multik-kotlin`, or `multik-openblas`),
      Kotlin version, and target platform (JVM, JS, Native, WASM).
    * Include the code that reproduces the problem. Provide the complete reproducer code, yet minimize it as much as
      possible.
    * However, don't put off reporting any weird or rarely appearing issues just because you cannot consistently
      reproduce them.
    * If the bug is in behavior, then explain what behavior you've expected and what you've got.
    * Include the full error message or stack trace.
* When submitting a feature request:
    * Use a 'feature request' template when creating a new issue.
    * Explain why you need the feature &mdash; what's your use-case, what's your domain.
    * Explaining the problem you face is more important than suggesting a solution.
      Even if you don't have a proposed solution, please report your problem.
    * If there is an alternative way to do what you need, then show the code of the alternative.

## Submitting PRs

We love PRs. Submit PRs [here](https://github.com/Kotlin/multik/pulls).
However, please keep in mind that maintainers will have to support the resulting code of the project,
so do familiarize yourself with the following guidelines.

* All development (both new features and bug fixes) is performed in the `develop` branch.
    * Base PRs against the `develop` branch.
    * Keep commits focused, one logical change per commit.
* If you fix documentation:
    * Documentation lives in [`docs/topics/`](docs/topics/).
    * API doc samples are in `multik-core/src/commonTest/kotlin/samples/docs/` and are validated
      by the [Korro](https://github.com/devcrocod/korro) plugin against markdown files in `docs/topics/`.
      Keep them in sync.
    * If you plan extensive rewrites/additions to the docs, then
      please [contact the maintainers](#contacting-maintainers)
      to coordinate the work in advance.
* If you make any code changes:
    * Follow the [Kotlin Coding Conventions](https://kotlinlang.org/docs/reference/coding-conventions.html).
        * Use 4 spaces for indentation.
        * Use imports with '*'.
    * [Build the project](#building) to make sure it all works and passes the tests.
* If you fix a bug:
    * Write the test that reproduces the bug.
    * Fixes without tests are accepted only in exceptional circumstances if it can be shown that writing the
      corresponding test is too hard or otherwise impractical.
    * Follow the style of writing tests that is used in this project:
      name test functions as `testXxx`. Don't use backticks in test names.
* If you introduce any new public APIs:
    * All new APIs must come with documentation and tests.
    * Follow the [Kotlin library authors' guidelines](https://kotlinlang.org/docs/api-guidelines-introduction.html).
      Key points for Multik:
        * Avoid data classes in public API (breaks binary compatibility when properties are added).
        * Don't add arguments to existing functions, even with defaults &mdash; use manual overloads instead.
        * Don't widen or narrow return types of existing functions.
        * Avoid boolean parameters &mdash; use separate named functions or enum types.
        * Avoid exposing mutable state &mdash; return read-only collections, make defensive copies of arrays.
        * Use `@RequiresOptIn` for experimental APIs, `@Deprecated` with progressive levels for removals.
    * If you plan large API additions, then please start by submitting an issue with the proposed API design
      to gather community feedback.
    * [Contact the maintainers](#contacting-maintainers) to coordinate any big piece of work in advance.
* If you propose/implement a new engine operation:
    * The `Math`, `LinAlg`, and `Statistics` interfaces in `multik-core` define the operations that each engine must
      implement.
    * Add the signature to the interface in `multik-core`.
    * Implement it in `multik-kotlin` (pure Kotlin, all platforms) and/or `multik-openblas` (native, JVM + desktop
      Native).
    * Add tests covering both engine implementations.
* Comment on the existing issue if you want to work on it. Ensure that the issue not only describes a problem,
  but also describes a solution that has received positive feedback. Propose a solution if there isn't any.

## Building

### Prerequisites

* **JDK 8+**.
* To build native OpenBLAS module: **gcc**, **g++**, and **gfortran** 8+ (same version).

### Build commands

```bash
# Build all modules (requires OpenBLAS toolchain)
./gradlew assemble

# Build without native OpenBLAS (if you don't have the C++ toolchain)
./gradlew assemble -x build_cmake

# Build only core module
./gradlew :multik-core:build
```

### Running tests

```bash
# JVM tests
./gradlew :multik-core:jvmTest

# Run a single JVM test class
./gradlew :multik-core:jvmTest --tests "org.jetbrains.kotlinx.multik.SomeTest"
```

### Generating API documentation

```bash
./gradlew dokkaGenerate
```

## Contacting maintainers

* If something cannot be done, not convenient, or does not work &mdash; submit an [issue](#submitting-issues).
* Discussions and general inquiries &mdash; use the `#datascience` channel in [Kotlin Slack](https://kotl.in/slack).
