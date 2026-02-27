---
name: multik-kdoc
description: "Write, update, and audit KDoc documentation for the Multik library. Handles the full cycle: KDoc comments on source code, syncing with Writerside user docs in docs/topics/, and creating/updating Korro code samples."
---

# Multik KDoc Documentation

Multik documentation lives at three layers. When documenting public API, always consider all three:

1. **KDoc comments** on source code
2. **Writerside docs** in `docs/topics/` — the user-facing documentation site
3. **Korro code samples** in `multik-core/src/commonTest/kotlin/samples/docs/` — executable `@Test` methods linked into the Writerside markdown

Korro samples exist to serve Writerside docs — they aren't needed for every KDoc, only for API elements that have (or should have) corresponding user documentation topics.

## Modes of Operation

### Targeted Documentation

The user points to a specific file, class, or function.

1. Read the source code. Understand behavior, edge cases, relationships with other API.
2. Write or update KDoc following the style guide below.
3. Search `docs/topics/` for references to this API element.
4. If user docs mention it, verify they still match. Update prose and code examples as needed.
5. If the doc topic uses Korro samples, verify and update the sample test files.
6. If no user doc exists for this element, ask the user whether to add a Writerside doc page (with Korro sample) or skip.

### Documentation Audit

The user asks to scan a module, package, or file for gaps.

1. Scan all declarations — `public`, `internal`, and `private`.
2. Flag elements with missing KDoc, empty KDoc (`/** */`), or KDoc that just restates the signature.
3. Skip `override` functions — project convention is to omit KDoc on overrides.
4. Present a summary: number of elements found, number undocumented, grouped by file.
5. Write KDoc for each, starting with the most visible/important. Public API gets full treatment; internal/private gets concise docs.
6. After writing public API KDoc, check user docs (steps 3-6 from Targeted flow).

## KDoc Style Guide

### Public API: Concise and Complete

Public KDoc should fit in a single IDE hover popup. This is the primary constraint — be thorough but not verbose.

**Structure (in this order):**

1. **Description** — what the element does. First sentence is the Dokka summary, make it count. Don't restate the signature.
2. **Example** (for functions) — a brief markdown-style code snippet showing typical usage.
3. **`@property`** / **`@param`** — one per property/parameter. Describe what it represents and valid range.
4. **`@return`** — when the return value isn't obvious from the name + type. Especially important for view-vs-copy semantics and shape changes.
5. **`@throws`** — document exceptions and the conditions that trigger them. Multik uses `require()` and `check()`, so `IllegalArgumentException` and `IllegalStateException` are common.
6. **`@see`** — link to 1-3 closely related functions for discoverability (e.g., `reshape` ↔ `flatten`). Don't overuse.

**Example — function with parameters:**

```kotlin
/**
 * Appends the given [value] elements to this array, returning a new flattened 1D array.
 *
 * The source array is flattened before appending. The original is not modified.
 *
 * ```
 * val a = mk.ndarray(mk[1, 2, 3])
 * a.append(4, 5) // [1, 2, 3, 4, 5]
 * ```
 *
 * @param value elements to append.
 * @return a new [D1Array] with size `this.size + value.size`.
 * @throws IllegalArgumentException if [value] contains elements incompatible with the array's [DataType].
 * @see [cat] for concatenation along a specific axis.
 */
public fun <T, D : Dimension> MultiArray<T, D>.append(vararg value: T): D1Array<T>
```

**Example — class:**

```kotlin
/**
 * Applies batched math operations to a [MutableMultiArray] in-place without allocating new arrays.
 *
 * Use the [math] block to chain operations sequentially:
 * ```
 * mk.math.inplace(array) {
 *     math { sin() }
 * }
 * ```
 *
 * @param T the numeric element type.
 * @param D the dimension type.
 * @param base the mutable array to modify.
 */
public open class InplaceOperation<T : Number, D : Dimension>(base: MutableMultiArray<T, D>)
```

### Internal / Private: Brief and Functional

For `internal` and `private` elements, keep KDoc concise — one or two sentences explaining *why* this exists and what it does. No `@param` tags unless parameters are non-obvious. No examples.

```kotlin
/** Computes strides from [shape] in row-major (C) order. Last dimension has stride 1. */
internal fun computeStrides(shape: IntArray): IntArray
```

```kotlin
/** Checks that [index] is within bounds for [axis] of size [size]. Throws [IndexOutOfBoundsException] if not. */
@PublishedApi
internal inline fun checkBounds(value: Boolean, index: Int, axis: Int, size: Int)
```

### General Rules

- Use `[ClassName]` and `[functionName]` for cross-references — they enable IDE navigation and Dokka links.
- Document view-vs-copy semantics explicitly. This is the #1 source of user confusion in ndarray libraries.
- For dimension-changing operations, describe the output shape.
- For generic type parameters (`T`, `D`), document constraints beyond the type bound only if they exist.
- Skip KDoc on `override` functions to avoid duplication.

## Syncing with Writerside Docs

After writing or updating KDoc on public API, check whether user docs need updating.

### Find Related Topics

Search `docs/topics/**/*.md` for the class/function name. Check `docs/mk.tree` for the topic hierarchy.

### Verify Consistency

If a topic references the element:
- Descriptions must match the KDoc (the code is the source of truth).
- Code examples must reflect current behavior.
- Parameter names and types must be correct in prose.

### Update Korro Samples

Korro is a build plugin that keeps code snippets in Writerside markdown in sync with actual Kotlin test code.

**How it works:** The gradle task `korro` scans markdown files listed in the `korro { docs = ... }` block of `multik-core/build.gradle.kts`. For each `<!---FUN name-->` ... `<!---END-->` block, it finds the matching `@Test fun name()` in sample test files and replaces the markdown code block with code between `// SampleStart` and `// SampleEnd`.

**Markdown side** (`docs/topics/`):

```markdown
<!---IMPORT samples.docs.userGuide.CreatingMultidimensionalArrays-->

## Literal Construction

<!---FUN literal_construction-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
// [1, 2, 3]
```

<!---END-->
```

**Kotlin side** (`multik-core/src/commonTest/kotlin/samples/docs/userGuide/`):

```kotlin
package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.*
import kotlin.test.Test

class CreatingMultidimensionalArrays {
    @Test
    fun literal_construction() {
        // SampleStart
        val a = mk.ndarray(mk[1, 2, 3])
        // [1, 2, 3]
        // SampleEnd
    }
}
```

Everything between `// SampleStart` and `// SampleEnd` is injected into the markdown — including output comments. This is intentional: output comments show users the expected result directly in the documentation.

**Output comment conventions:**
- Single-line output: inline comment on the same line — `a[2] // 3`
- Multi-line output: block comment `/* ... */` below the expression
- Sometimes `println()` + output comment to be explicit

**Rules:**
- `<!---FUN name-->` must match the test function name exactly.
- `<!---IMPORT package.ClassName-->` must match the test class FQN.
- Samples must compile and pass — they are `@Test` methods.
- User guide samples go in `samples/docs/userGuide/`, API reference in `samples/docs/apiDocs/`.

**When to create/update samples:**
- When documenting a public API function that has a Writerside topic with `<!---FUN-->` blocks.
- When the existing sample code doesn't match updated behavior.
- When adding a new section to a Writerside topic that needs an executable example.

### Validate

After updating samples, remind the user to run:

```bash
./gradlew :multik-core:jvmTest --tests "samples.docs.*"
```
