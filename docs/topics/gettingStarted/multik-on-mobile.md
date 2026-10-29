# Multik on mobile (Android, iOS)

<web-summary>
Use Multik on Android and iOS with Kotlin Multiplatform.
Learn which engines are available and what limitations to expect on mobile targets.
</web-summary>

<card-summary>
Multik runs on Android and iOS.
On Android, OpenBLAS is available on arm64-v8a devices; iOS uses the Kotlin engine.
</card-summary>

<link-summary>
Engine availability and KMP setup for Multik on Android and iOS.
</link-summary>

## Overview

Multik can be used in mobile apps via Kotlin Multiplatform:

- Android - Android target (AAR artifacts, minSdk 28).
- iOS - Kotlin/Native target.

The recommended engine for mobile is **KOTLIN**, which provides the broadest compatibility.
On Android, **DEFAULT** also uses OpenBLAS on `arm64-v8a` devices.

## Engine availability

| Target       | Engines                            | Notes                                                                 |
|--------------|------------------------------------|-----------------------------------------------------------------------|
| Android      | DEFAULT, KOTLIN, NATIVE (OpenBLAS) | OpenBLAS supports `arm64-v8a` only; on other ABIs DEFAULT uses KOTLIN. |
| iOS (Native) | KOTLIN                             | OpenBLAS is not available.                                            |

## Dependencies

A typical KMP setup:

- Add `multik-core` to `commonMain`.
- Add `multik-kotlin` (or `multik-default`) to the Android and iOS source sets.

See [](installation.md) for Gradle examples.

> **Notes:**
>
> With `multik-openblas` alone, the NATIVE engine throws `EngineMultikException` on ABIs other than `arm64-v8a`;
> `multik-default` falls back to the Kotlin engine instead.
> For iOS, the Kotlin engine is the only supported option.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="multik-on-different-platforms.md" summary="Overview of Multik targets and engine availability."/>
  <a href="engines-of-multik.md" summary="Understand DEFAULT, KOTLIN, and NATIVE engines."/>
</category>
<category ref="get-start">
  <a href="installation.md" summary="Add Multik dependencies to your project."/>
  <a href="multik-on-different-platforms.md" summary="Full list of Kotlin target presets supported by Multik."/>
</category>
</seealso>
