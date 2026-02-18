# Multik on WASM

<web-summary>
Use Multik in Kotlin/WASM projects with the pure Kotlin engine.
Learn how to set dependencies and where to find platform-specific notes.
</web-summary>

<card-summary>
Multik works on WASM via the Kotlin engine.
Keep the same NDArray API across JVM, JS, Native, and WASM.
</card-summary>

<link-summary>
Use Multik on Kotlin/WASM with the Kotlin engine.
</link-summary>

## Overview

Multik supports Kotlin/WASM with the **KOTLIN** engine.
This allows you to use ndarrays in WebAssembly environments while keeping the same API surface.

## Engine availability

- **KOTLIN** only. `NATIVE` (OpenBLAS) is not available on WASM.

## Dependencies

For Kotlin/WASM projects:

- Use `multik-core` in `commonMain`.
- Add `multik-kotlin` (or `multik-default`) for the WASM target.

See [](installation.md) for Gradle examples.

> **Notes:**
> 
> For heavy computation, consider running the workload on JVM or Native and exchanging results over an API.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="multik-on-JavaScript.md" summary="Run Multik in Kotlin/JS with the Kotlin engine."/>
  <a href="multik-on-different-platforms.md" summary="Overview of Multik targets and engine availability."/>
  <a href="engines-of-multik.md" summary="Understand DEFAULT, KOTLIN, and NATIVE engines."/>
</category>
<category ref="get-start">
  <a href="installation.md" summary="Add Multik dependencies to your project."/>
  <a href="multik-on-different-platforms.md" summary="Full list of Kotlin target presets supported by Multik."/>
</category>
</seealso>
