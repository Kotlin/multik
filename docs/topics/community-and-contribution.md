# Community & Contribution

<web-summary>
How to get help with Multik, report issues, and contribute — GitHub, Kotlin Slack, and contribution guidelines.
</web-summary>

<card-summary>
Get help, report issues, and contribute to Multik.
</card-summary>

<link-summary>
Community & contribution — get help, report issues, contribute.
</link-summary>

## Project info

Multik is a [JetBrains incubator](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub) project
licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

- **GitHub:** [github.com/Kotlin/multik](https://github.com/Kotlin/multik)
- **Maven Central:** `org.jetbrains.kotlinx:multik-core`
- **Stability:** Alpha

## Getting help

### GitHub Issues

The primary place to ask questions, report bugs, and request features is
[GitHub Issues](https://github.com/Kotlin/multik/issues).

Before opening a new issue:

1. Search existing issues — your question may already be answered.
2. Check the [FAQ](FAQ.md) for common problems and workarounds.

When reporting a bug, include:

- Multik version and module (`multik-default`, `multik-kotlin`, or `multik-openblas`).
- Kotlin version and target platform (JVM, JS, Native, WASM, Android).
- A minimal code sample that reproduces the problem.
- The full error message or stack trace.

### Kotlin Slack

Join the [Kotlin Slack](https://kotlinlang.slack.com) workspace
([request an invite](https://surveys.jetbrains.com/s3/kotlin-slack-sign-up)) and look for
discussions in the **#datascience** channel.

### Stack Overflow

Tag your questions with **kotlin** and **multik** for community visibility.

## Contributing

Contributions are welcome — bug fixes, new features, documentation improvements, and tests.

### Setting up the development environment

1. **Fork** the repository on GitHub and clone your fork.
2. To build all modules including native code, you also need **gcc**, **g++**, and **gfortran** 8+
   (same version), and the `JAVA_HOME` environment variable set.

```bash
# Build everything
./gradlew assemble

# Build without native OpenBLAS (if you don't have the C++ toolchain)
./gradlew assemble -x build_cmake

# Run JVM tests
./gradlew :multik-core:jvmTest
```

### Ways to contribute

**Report bugs**
: Open an [issue](https://github.com/Kotlin/multik/issues/new) with a clear description
and a minimal reproducer.

**Implement a missing API**
: The `Math`, `LinAlg`, and `Statistics` interfaces define the operations that each engine must
implement. You can add a new function by:
1. Adding the signature to the interface in `multik-core`.
2. Implementing it in `multik-kotlin` (pure Kotlin) and/or `multik-openblas` (native).

When adding or modifying public API, follow the
[Kotlin library authors' guidelines](https://kotlinlang.org/docs/api-guidelines-introduction.html).
Key points: always specify explicit visibility and return types (explicit API mode is enabled),
avoid data classes in public API, don't add arguments to existing functions (use overloads),
avoid boolean parameters, don't expose mutable state, and use `@RequiresOptIn` for experimental APIs.

**Create a custom engine**
: Subclass `Engine` and implement the `Math`, `LinAlg`, and `Statistics` interfaces. Register it
at runtime with `mk.addEngine()` or `mk.setEngine()`.

**Improve documentation**
: Documentation lives in `docs/topics/` (Writerside format). API doc samples are in
`multik-core/src/commonTest/kotlin/samples/docs/` and validated by the Korro plugin.

**Add tests**
: Tests are always welcome. Place JVM tests under `src/jvmTest/`, common tests under
`src/commonTest/`. Run them with `./gradlew :multik-core:jvmTest` (or the corresponding
target task).

### Submitting a pull request

1. Create a branch from `develop`.
2. Keep commits focused — one logical change per commit.
3. Add or update tests for any new behavior.
4. Make sure `./gradlew :multik-core:jvmTest` passes before submitting.
5. Open a pull request against the `develop` branch with a clear description of the change.

## Roadmap and open feature requests

The following features are frequently requested and tracked on GitHub:

| Feature                       | Issue                                               |
|-------------------------------|-----------------------------------------------------|
| Boolean masks and `where`     | [#220](https://github.com/Kotlin/multik/issues/220) |
| Broadcasting / scalar ops     | [#219](https://github.com/Kotlin/multik/issues/219) |
| Zero-sized ndarrays           | [#217](https://github.com/Kotlin/multik/issues/217) |
| Element-wise `power` function | [#162](https://github.com/Kotlin/multik/issues/162) |
| Multiplatform CSV             | [#158](https://github.com/Kotlin/multik/issues/158) |
| Platform-specific artifacts   | [#148](https://github.com/Kotlin/multik/issues/148) |
| Sparse matrix support         | [#37](https://github.com/Kotlin/multik/issues/37)   |

<seealso style="cards">
<category ref="get-start">
  <a href="installation.md" summary="How to add Multik to your project."/>
  <a href="engines-of-multik.md" summary="Engines and how to choose between them."/>
</category>
<category ref="ext">
  <a href="FAQ.md" summary="Frequently asked questions."/>
</category>
</seealso>
