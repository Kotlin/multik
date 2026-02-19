# Multik on mobile (Android, iOS)

<web-summary>
Use Multik on Android and iOS with Kotlin Multiplatform.
Learn which engines are available and what limitations to expect on mobile targets.
</web-summary>

<card-summary>
Multik runs on Android and iOS via the Kotlin engine.
OpenBLAS is limited and platform-dependent on mobile.
</card-summary>

<link-summary>
Use Multik on Android and iOS with the Kotlin engine and KMP setup.
</link-summary>

## Overview

Multik can be used in mobile apps via Kotlin Multiplatform:

- Android - JVM target.
- iOS - Kotlin/Native target.

The recommended engine for mobile is **KOTLIN**, which provides the broadest compatibility.

## Engine availability

| Target        | Engines                   | Notes                             |
|---------------|---------------------------|-----------------------------------|
| Android (JVM) | KOTLIN; NATIVE (OpenBLAS) | OpenBLAS supports arm64-v8a only. |
| iOS (Native)  | KOTLIN                    | OpenBLAS is not available.        |

## Dependencies

A typical KMP setup:

- Add `multik-core` to `commonMain`.
- Add `multik-kotlin` (or `multik-default`) to the Android and iOS source sets.

See [](installation.md) for Gradle examples.

> **Notes:**
>
> Use OpenBLAS on Android only if you control the ABI and need the extra performance.
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
