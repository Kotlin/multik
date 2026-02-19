# Multik on different platforms

<web-summary>
Learn how Multik runs across JVM, JS, Native, and WASM.
Understand which engines are available on each platform and where to find platform-specific guidance.
</web-summary>

<card-summary>
Multik is multiplatform by design.
See how engines map to JVM, JS, Native, and WASM targets.
</card-summary>

<link-summary>
See how Multik works across JVM, JS, Native, and WASM targets.
</link-summary>

## Overview

Multik is a Kotlin Multiplatform library.
You can use the same NDArray API across JVM, JS, Native, and WASM,
and choose an execution engine that fits your target and performance needs.

## Supported targets

Multik supports the major Kotlin targets:

- JVM
- JS
- Native (iOS, macOS, Linux, Windows)
- WASM

### Target presets

The following [target presets](https://kotlinlang.org/docs/multiplatform-dsl-reference.html) are supported:

| Target platform | Target presets                            |
|-----------------|-------------------------------------------|
| Kotlin/JVM      | `jvm`                                     |
| iOS             | `iosArm64`, `iosX64`, `iosSimulatorArm64` |
| macOS           | `macosX64`, `macosArm64`                  |
| Linux           | `linuxX64`                                |
| Windows         | `mingwX64`                                |
| JS              | `js`                                      |
| WASM            | `wasm`                                    |

## Engine availability by platform

Multik separates the API from execution engines:

- **DEFAULT** chooses the best available engine at runtime.
- **KOTLIN** is a pure Kotlin implementation and is the safest choice for portability.
- **NATIVE** uses OpenBLAS for higher performance where available.

| Target           | Engines                                       | Notes                                                                                                                |
|------------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| JVM              | DEFAULT, KOTLIN, NATIVE (OpenBLAS)            | NATIVE requires the OpenBLAS artifact.                                                                               |
| JS               | KOTLIN                                        | Browser and Node.js targets.                                                                                         |
| WASM             | KOTLIN                                        | Kotlin/WASM targets.                                                                                                 |
| Native (desktop) | KOTLIN; NATIVE (OpenBLAS) on selected targets | OpenBLAS for desktop native targets (`linuxX64`, `mingwX64`, `macosX64`, `macosArm64`) is experimental and unstable. |
| Native (iOS)     | KOTLIN                                        | OpenBLAS is not available.                                                                                           |

For details such as Android ABI limits, see the platform pages below.

## Multiplatform setup pattern

A common approach is:

- Add `multik-core` to `commonMain`.
- Add engine dependencies per target: `multik-kotlin` for JS/WASM/Native, and `multik-default` or `multik-openblas` for
  JVM.

For dependency examples, see [](installation.md).

## Platform-specific guides

- [](multik-on-JVM.md)
- [](multik-on-JavaScript.md)
- [](multik-on-WASM.md)
- [](multik-on-mobile.md)
- [](multik-on-desktop.md)

If you spot a typo or missing detail, please [open a docs issue](https://github.com/Kotlin/multik/issues/new/choose).

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="engines-of-multik.md" summary="Understand DEFAULT, KOTLIN, and NATIVE engines."/>
  <a href="multik-on-JVM.md" summary="JVM-specific engine choices and setup notes."/>
  <a href="multik-on-JavaScript.md" summary="Run Multik in Kotlin/JS with the Kotlin engine."/>
  <a href="multik-on-WASM.md" summary="Use Multik in Kotlin/WASM targets."/>
  <a href="multik-on-mobile.md" summary="Android and iOS usage notes."/>
  <a href="multik-on-desktop.md" summary="Desktop JVM/Native guidance and OpenBLAS notes."/>
</category>
<category ref="get-start">
  <a href="installation.md" summary="Add Multik dependencies to your project."/>
</category>
</seealso>
