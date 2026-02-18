# Multik on JavaScript

<web-summary>
Run Multik in Kotlin/JS projects with the pure Kotlin engine.
Learn which dependencies to use and where to find platform-specific details.
</web-summary>

<card-summary>
Multik works on Kotlin/JS using the Kotlin engine.
Use the same NDArray API in browser or Node.js targets.
</card-summary>

<link-summary>
Use Multik in Kotlin/JS with the Kotlin engine.
</link-summary>

## Overview

Multik supports Kotlin/JS with the **KOTLIN** engine.
This provides a consistent NDArray API for browser and Node.js targets.

## Engine availability

- **KOTLIN** only. `NATIVE` (OpenBLAS) is not available on JS.

## Dependencies

For Kotlin/JS projects:

- Use `multik-core` in `commonMain`.
- Add `multik-kotlin` (or `multik-default`) for the JS target.

See [](installation.md) for Gradle examples.

> **Notes:**
> - Prefer smaller array sizes for interactive or UI-driven work.
> - For heavy computation, consider running the workload on JVM or Native.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="multik-on-WASM.md" summary="Use Multik in Kotlin/WASM targets."/>
  <a href="multik-on-different-platforms.md" summary="Overview of Multik targets and engine availability."/>
  <a href="engines-of-multik.md" summary="Understand DEFAULT, KOTLIN, and NATIVE engines."/>
</category>
<category ref="get-start">
  <a href="installation.md" summary="Add Multik dependencies to your project."/>
  <a href="multik-on-different-platforms.md" summary="Full list of Kotlin target presets supported by Multik."/>
</category>
</seealso>
