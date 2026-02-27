# API Reference Templates

Strict structural templates for API Reference pages in `docs/topics/apiDocs/`. All API reference pages must follow these templates to ensure uniform appearance.

## Table of Contents

- [Single Function Page](#single-function)
- [Multi-Function Page](#multi-function)
- [Index Page](#index-page)

---

## Single Function Page {#single-function}

The most common API reference page. One function (possibly with overloads) per page.

### Template

```markdown
# functionName

<!---IMPORT samples.docs.apiDocs.SampleClassName-->

<web-summary>
API reference for mk.functionName() — what it does and when to use it.
</web-summary>

<card-summary>
One sentence about the function's purpose.
</card-summary>

<link-summary>
mk.functionName() — very brief description.
</link-summary>

1-3 sentences describing what the function does, when you'd use it, and any key behavior
the reader should know upfront. This description appears before the signature and gives
context that the summaries (which are metadata-only and not rendered on the page) cannot.

## Signature

```kotlin
inline fun <reified T : Any> Multik.functionName(
    param1: Type1,
    param2: Type2
): ReturnType
```

## Parameters

| Parameter | Type    | Description                                          |
|-----------|---------|------------------------------------------------------|
| `param1`  | `Type1` | What this parameter controls. Constraints if any.    |
| `param2`  | `Type2` | What this parameter controls.                        |

**Returns:** `ReturnType` — brief description of what is returned.

## Example

<!---FUN functionName_example-->

```kotlin
val result = mk.functionName(arg1, arg2)
// expected output as comment
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="related1.md" summary="Brief description."/>
  <a href="related2.md" summary="Brief description."/>
</category>
</seealso>
```

### Rules

- Every single-function page must have a **description paragraph** between the summaries and `## Signature`. This is the reader's first impression — explain what the function does and the main use case. The summaries are metadata (not shown on the page), so the description is the first visible prose.
- `## Signature` (singular) for one signature; `## Signatures` (plural) for overloaded functions
- For overloads, list all signatures in one code block separated by blank lines
- Parameters table columns: Parameter | Type | Description. Add `Default` column only when some parameters have defaults
- `**Returns:**` is bold, on its own line after the parameters table
- Add `**Throws:**` line if the function throws specific exceptions (e.g., `IllegalArgumentException`)
- Optional sections at the end: `## Constraints` or `## Pitfalls` — only for non-obvious gotchas
- If the API is experimental, add a warning admonition before the Signature section:

```markdown
> This is an experimental API. It may be changed or removed in future releases.
> Requires `@OptIn(ExperimentalMultikApi::class)`.
> {style="warning"}
```

### Real example: d1array.md

```markdown
# d1array

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.d1array() — create a 1D NDArray with an init lambda.
</web-summary>

<card-summary>
Create a 1D array where each element is computed by an init function.
</card-summary>

<link-summary>
mk.d1array() — 1D array from init lambda.
</link-summary>

Creates a 1-dimensional array of the given size, where each element is computed
by an init lambda that receives the element index. Use this when you need
fine-grained control over element values during array construction.

## Signature

```kotlin
inline fun <reified T : Any> Multik.d1array(
    sizeD1: Int,
    noinline init: (Int) -> T
): D1Array<T>
```

## Parameters

| Parameter | Type         | Description                                                                       |
|-----------|--------------|-----------------------------------------------------------------------------------|
| `sizeD1`  | `Int`        | Number of elements. Must be positive.                                             |
| `init`    | `(Int) -> T` | Lambda that receives the index (0 to `sizeD1 - 1`) and returns the element value. |

**Returns:** `D1Array<T>`

## Example

<!---FUN d1array_example-->

```kotlin
val a = mk.d1array<Int>(5) { it * 2 }
// [0, 2, 4, 6, 8]

val b = mk.d1array<Double>(3) { it + 0.5 }
// [0.5, 1.5, 2.5]
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="d2array.md" summary="Create 2D arrays with a flat-index init lambda."/>
  <a href="ndarray.md" summary="Create arrays from lists and collections."/>
  <a href="zeros.md" summary="Create zero-filled arrays."/>
</category>
</seealso>
```

---

## Multi-Function Page {#multi-function}

Used for closely related functions documented on a single page.

### Template

```markdown
# Page Title

<web-summary>
API reference for ... — list the functions covered.
</web-summary>

<card-summary>
Brief summary of the function group.
</card-summary>

<link-summary>
Short summary of the function group.
</link-summary>

## Overview

Brief paragraph explaining what these functions have in common.

| Function    | Description                      |
|-------------|----------------------------------|
| `function1` | One-line description.            |
| `function2` | One-line description.            |

## function1

{id="function1"}

1-3 sentences describing what this function does and its main use case.

### Signatures

{id="function1-signatures"}

```kotlin
fun <T : Number, D : Dimension> Interface.function1(a: MultiArray<T, D>): ReturnType
```

### Parameters

{id="function1-params"}

| Parameter | Type               | Description  |
|-----------|--------------------|--------------|
| `a`       | `MultiArray<T, D>` | Input array. |

**Returns:** `ReturnType`

### Example

{id="function1-example"}

```kotlin
val result = mk.stat.function1(array)
// expected output
```

## function2

{id="function2"}

... same nested structure (Signatures → Parameters → Example) ...

## Pitfalls

* Gotcha for function1.
* Gotcha for function2.

<seealso style="cards">
<category ref="api-docs">
  <a href="related.md" summary="Description."/>
</category>
</seealso>
```

### Rules

- Each function gets its own H2 heading with `{id="function-name"}` anchor on a **blank line below** the heading
- Nested H3 sections each get their own anchor: `{id="function-name-signatures"}`, `{id="function-name-params"}`, `{id="function-name-example"}`
- Use `### Signature` (singular) vs `### Signatures` (plural) based on overload count
- Shared `## Pitfalls` section at the end covers gotchas across all functions
- The overview table at the top provides a quick summary of all functions on the page

### Anchor ID format

```markdown
## mean

{id="mean"}

### Signatures

{id="mean-signatures"}

### Parameters

{id="mean-params"}

### Example

{id="mean-example"}
```

The `{id="..."}` attribute goes on its own blank line immediately after the heading. This enables deep linking from other pages: `[`mean`](descriptive-statistics.md#mean)`.

---

## Index Page {#index-page}

Groups related function pages into a navigable overview. No code examples — individual function pages have those.

### Template

```markdown
# Category Name

<web-summary>
API reference for all Multik ... functions: list key function names.
</web-summary>

<card-summary>
All functions for ... — brief scope description.
</card-summary>

<link-summary>
Category reference — key function names.
</link-summary>

## Overview

One paragraph explaining this group of functions and how to access them (via `mk`, `mk.math`, `mk.stat`, `mk.linalg`).

### Sub-group 1

| Function          | Description                                   |
|-------------------|-----------------------------------------------|
| [](function1.md)  | One-line description.                          |
| [](function2.md)  | One-line description.                          |

### Sub-group 2

| Function          | Description                                   |
|-------------------|-----------------------------------------------|
| [](function3.md)  | One-line description.                          |

<seealso style="cards">
<category ref="user-guide">
  <a href="related-guide.md" summary="Description."/>
</category>
</seealso>
```

### Rules

- Use `[](page.md)` auto-titled links in tables for consistency
- Group functions by usage pattern or conceptual similarity
- One-line descriptions per function — keep it scannable
- No Korro samples on index pages
- The `## Overview` paragraph should mention the entry point object (e.g., "All creation functions are called on the `Multik` object (aliased as `mk`)")
