# Multik on desktop (MacOS, Win, Linux)

<web-summary>
Use Multik on desktop platforms (Windows, macOS, Linux) via JVM or Native targets.
Choose between the pure Kotlin engine and OpenBLAS where available.
</web-summary>

<card-summary>
Multik works on desktop with JVM and Native targets.
Use KOTLIN for portability or NATIVE for speed where supported.
</card-summary>

<link-summary>
Run Multik on desktop platforms with JVM/Native targets and the right engine.
</link-summary>

## Overview

Desktop use cases typically fall into two categories:

- JVM desktop apps (Swing/JavaFX/Compose for Desktop).
- Kotlin/Native desktop binaries (macOS, Linux, Windows).

Both can use the same Multik API, with engine choice depending on platform and performance needs.

## Engine availability

| Target                                                            | Engines                            | Notes                                                             |
|-------------------------------------------------------------------|------------------------------------|-------------------------------------------------------------------|
| JVM desktop                                                       | DEFAULT, KOTLIN, NATIVE (OpenBLAS) | NATIVE requires `multik-openblas`.                                |
| Native desktop (`linuxX64`, `mingwX64`, `macosX64`, `macosArm64`) | KOTLIN; NATIVE (OpenBLAS)          | OpenBLAS for desktop native targets is experimental and unstable. |

## Dependencies

- Use `multik-default` for convenience on the JVM desktop.
- Use `multik-kotlin` for Kotlin/Native or when you want to avoid native dependencies.
- Use `multik-openblas` when you need performance and your target supports it.

See [](installation.md) for dependency examples.

> **Notes:**
>
> If you depend on OpenBLAS, test on all target machines to validate compatibility and performance.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="multik-on-JVM.md" summary="JVM-specific engine choices and setup notes."/>
  <a href="multik-on-different-platforms.md" summary="Overview of Multik targets and engine availability."/>
  <a href="engines-of-multik.md" summary="Understand DEFAULT, KOTLIN, and NATIVE engines."/>
</category>
<category ref="get-start">
  <a href="installation.md" summary="Add Multik dependencies to your project."/>
  <a href="multik-on-different-platforms.md" summary="Full list of Kotlin target presets supported by Multik."/>
</category>
</seealso>
