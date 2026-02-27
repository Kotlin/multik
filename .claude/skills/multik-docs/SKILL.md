---
name: multik-docs
description: "Write and edit Writerside user documentation pages for the Multik library. Covers all page types: Getting Started, User Guide, and API Reference. Handles page structure, Korro code samples, mk.tree navigation, and cross-references. Use this skill when creating new doc pages, updating existing ones, or when the multik-kdoc skill needs to create/update Writerside topics."
---

# Multik User Documentation

Write and edit Writerside documentation for the Multik library. Documentation lives in `docs/topics/` and is organized into three categories, each with distinct structure and purpose.

## Documentation Philosophy

Before writing, internalize these principles:

**From the Diataxis framework** (full reference: `references/diataxis-framework.md`):
- Tutorials teach beginners through step-by-step instructions with concrete results
- Guides solve specific problems for intermediate users
- Reference docs provide precise technical information for lookup
- Explanations build deeper understanding for experts

**Project doc rules** (full reference: `references/doc-quality-rules.md`):
- Be concise — make every token count
- Optimize for skimming: bold, lists, headings, code examples
- Keep the first-time experience simple, reveal complexity gradually
- Ensure docs have many code examples users can copy/paste or run
- Cross-link related guides, APIs, and examples
- Make content easy for AI to consume as Markdown

## Page Types

### API Reference (`docs/topics/apiDocs/`) — strict templates

**Audience:** Experienced users. **Purpose:** Precise technical lookup.

API reference pages must look uniform. Three sub-types:
- **Single function pages** — one function (possibly overloaded) per page: Signature → Parameters → Returns → Example → optional Pitfalls
- **Multi-function pages** — for closely related functions on one page (e.g., `descriptive-statistics.md`): each function gets its own H2 with nested H3 sub-sections and anchor IDs
- **Index pages** — group related function pages into navigable tables (e.g., `array-creation.md`, `statistics.md`)

Read `references/api-reference-templates.md` for exact structural templates with real examples.

### Free-form pages — no rigid template

These pages vary in structure depending on the topic. Instead of following a template, **read 2-3 existing pages of the same type** before writing to match the tone and patterns.

**Getting Started** (`docs/topics/gettingStarted/`): Narrative style for beginners. Onboarding, installation, platform setup. May use tabs for Gradle Kotlin/Groovy, bullet lists, multiple Korro imports. Examples: `quickstart.md`, `installation.md`, `engines-of-multik.md`.

**User Guide** (`docs/topics/userGuide/`): Task-oriented for intermediate users. Typically opens with `## Overview`, each section is self-contained (brief prose + code example). Examples: `copies-and-views.md`, `indexing-and-slicing.md`, `creating-multidimensional-arrays.md`.

**FAQ** (`docs/topics/FAQ.md`): H2 headings for topic groups, H3 headings as questions. Direct answers with code examples, uses **Fix:** and **Workaround:** labels.

**Community & Contribution** (`docs/topics/community-and-contribution.md`): Freeform informational page with links, lists, tables.

**Landing pages** (`.topic` XML files): Pure XML format for section entry points (e.g., `welcome.topic`, `getting-started.topic`). See `references/writerside-markup.md` for the XML schema.

## Workflow

### Creating a New Page

1. **Determine the page type** based on audience and purpose (see above).
2. **For API Reference:** read `references/api-reference-templates.md` and follow the exact template for the sub-type (only load this file for API Reference pages). **For other pages:** read 2-3 existing pages of the same type to match tone and structure. Look at nearby pages in `mk.tree` for context.
3. **Read source code** — understand the API you're documenting. Check KDoc for descriptions, parameters, edge cases.
4. **Write the page** — API Reference uses strict templates; other pages adapt to their content while following the common elements below.
5. **Create Korro samples** if the page needs executable code examples (see Korro section below).
6. **Update `docs/mk.tree`** to add the page to the navigation (see mk.tree section below).
7. **Add cross-references** — link from existing related pages to the new one, and from the new page to related ones via `<seealso>` and inline links.

### Editing an Existing Page

1. **Read the current page** and understand its structure and content.
2. **Read the source code** for the API being documented to check accuracy.
3. **Make targeted changes** — don't restructure the whole page unless asked.
4. **Update Korro samples** if code examples changed (both the markdown block and the test file).
5. **Check cross-references** — update any pages that link to changed sections.

## Common Page Elements

Every markdown page uses these elements in this exact order at the top:

```markdown
# Page Title

<!---IMPORT samples.docs.category.ClassName-->

<web-summary>
Two to three sentences for search results and SEO.
</web-summary>

<card-summary>
One to two sentences for card previews.
</card-summary>

<link-summary>
One sentence for inline link hover previews.
</link-summary>
```

- `# Title` is always the first line (H1). Exactly one per page.
- `<!---IMPORT ...-->` appears only when the page uses Korro samples. Multiple imports are allowed.
- The three summary blocks are required on every page, always in web → card → link order.
- `web-summary` is the longest, `link-summary` is the shortest.

### Seealso Block

Every page ends with a `<seealso>` block for related content:

```xml
<seealso style="cards">
<category ref="api-docs">
  <a href="page-name.md" summary="Brief description of the linked page."/>
</category>
</seealso>
```

Or with a title and multiple categories:

```xml
<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="page.md" summary="Description."/>
</category>
<category ref="api-docs">
  <a href="page.md" summary="Description.">Custom Link Title</a>
</category>
</seealso>
```

Category refs: `"user-guide"`, `"api-docs"`, `"get-start"`, `"ext"` (external links).

### Cross-References

Three patterns, in order of preference:

1. **Auto-titled link:** `[](page-name.md)` — resolves to the page title. Best for inline references.
2. **Custom text link:** `[link text](page-name.md)` — for when the page title doesn't fit the context.
3. **Deep link with anchor:** `[text](page-name.md#anchor-id)` — for linking to specific sections.

### Admonitions

```markdown
> Warning text here.
> {style="warning"}

> Note text here.
> {style="note"}

> Tip text here.
> {style="tip"}
```

When using non-standard Writerside elements (tabs, compare blocks, procedures, topic switchers, etc.), read `references/writerside-markup.md` for syntax reference. Skip this file for pages that only use standard markdown, code blocks, admonitions, and seealso blocks.

## Korro Code Samples

Korro is a Gradle plugin that synchronizes code snippets between Writerside markdown and executable Kotlin test files. This ensures documentation examples always compile and produce expected output.

### How It Works

**Markdown side** (`docs/topics/{category}/page.md`):

```markdown
<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<!---FUN arange_example-->

```kotlin
val a = mk.arange<Int>(5)              // [0, 1, 2, 3, 4]
val b = mk.arange<Int>(2, 10, 3)       // [2, 5, 8]
```

<!---END-->
```

**Kotlin side** (`multik-core/src/commonTest/kotlin/samples/docs/{category}/ClassName.kt`):

```kotlin
package samples.docs.apiDocs

import org.jetbrains.kotlinx.multik.api.*
import kotlin.test.Test

class ArrayCreation {
    @Test
    fun arange_example() {
        // SampleStart
        val a = mk.arange<Int>(5)              // [0, 1, 2, 3, 4]
        val b = mk.arange<Int>(2, 10, 3)       // [2, 5, 8]
        // SampleEnd
    }
}
```

### Key Rules

- `<!---FUN name-->` must match the `@Test fun name()` method name exactly
- `<!---IMPORT package.ClassName-->` must match the test class fully-qualified name
- Code between `// SampleStart` and `// SampleEnd` is what appears in the markdown code block
- Assertions (`assertEquals(...)`) go after `// SampleEnd` — they validate but don't appear in docs
- Output is shown as comments: single-line with `//`, multi-line with `/* ... */`
- Sample files are organized by category:
  - `samples/docs/apiDocs/` for API Reference pages
  - `samples/docs/userGuide/` for User Guide pages
  - `samples/docs/` root for Getting Started and other pages

### Creating a New Sample

1. Check if an existing sample class covers the topic. If so, add a new `@Test fun` method to it.
2. If not, create a new class. Name it in PascalCase matching the topic (e.g., `ShapeManipulation.kt` for `shape-manipulation.md`).
3. Package must match the directory: `samples.docs.apiDocs` or `samples.docs.userGuide`.
4. Add the `<!---IMPORT ...-->` directive at the top of the markdown page.
5. Wrap each code example in `<!---FUN name-->` / `<!---END-->` blocks in the markdown.

### Validation

After creating or updating samples, remind the user to run:

```bash
./gradlew :multik-core:jvmTest --tests "samples.docs.*"
```

## Updating mk.tree

The navigation tree is defined in `docs/mk.tree`. When adding a new page:

1. Find the correct parent `<toc-element>` for the page type:
   - Getting Started: inside `<toc-element toc-title="Getting started" topic="getting-started.topic">`
   - User Guide: inside `<toc-element topic="user-guide.topic">`
   - API Reference: inside `<toc-element toc-title="API reference">`
2. Add a `<toc-element topic="your-page.md"/>` at the appropriate position.
3. For sub-pages, nest inside the parent `<toc-element>`.
4. Use `toc-title="Custom Title"` only when the page's H1 isn't suitable for the sidebar.
5. Use `wip="true"` for pages that are not yet complete.

Example:
```xml
<toc-element topic="array-creation.md">
    <toc-element topic="arange.md"/>
    <toc-element topic="your-new-function.md"/>
    <!-- ... -->
</toc-element>
```

## Writing Quality Checklist

Before finishing any page:

- [ ] H1 title is present and descriptive
- [ ] All three summary blocks are present (web, card, link)
- [ ] Code examples compile and show expected output
- [ ] Korro directives match the sample test files (`<!---IMPORT-->` and `<!---FUN-->`)
- [ ] Cross-references use correct page filenames
- [ ] `<seealso>` block at the end with 2-4 relevant links
- [ ] Page added to `mk.tree` in the correct location (for new pages)
- [ ] Tone matches the page type (tutorial vs reference vs guide)
- [ ] No jargon or idioms without explanation
- [ ] Optimized for skimming — short paragraphs, bold key terms, lists, headings
