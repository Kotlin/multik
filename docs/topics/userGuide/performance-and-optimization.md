# Performance and optimization

<!---IMPORT samples.docs.userGuide.PerformanceAndOptimization-->

<web-summary>
Learn practical tips for efficient Multik usage: choose the right engine, avoid unnecessary copies,
and work with contiguous data when performance matters.
</web-summary>

<card-summary>
Optimize Multik by selecting the right engine and reducing allocations.
Prefer contiguous data and in-place operations when safe.
</card-summary>

<link-summary>
Performance tips for Multik: engine choice, copies vs views, and allocation control.
</link-summary>

## Overview

Multik performance depends on three main factors:

- **Engine** (KOTLIN vs NATIVE/OpenBLAS)
- **Data layout** (contiguous arrays vs views)
- **Allocation patterns** (temporary arrays and conversions)

This section highlights practical guidelines for common performance pitfalls.

## Choose the right engine

- **DEFAULT** picks the best available engine automatically.
- **KOTLIN** is portable and works everywhere.
- **NATIVE** uses OpenBLAS for faster linear algebra on supported platforms.

See [Engines of Multik](engines-of-multik.md) for selection details.

## Prefer contiguous data

Multik uses a contiguous primitive array under the hood.
Views (created by slicing or indexing) may be non-contiguous and can be slower to iterate.

Use `deepCopy()` to materialize a contiguous array when needed:

<!---FUN deepCopy_example-->

```kotlin
val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
val v = a[0] // view

println("v.consistent is ${v.consistent}") // false

val c = v.deepCopy()
println("c.consistent is ${c.consistent}") // true
```

<!---END-->

`flatten()` also returns a contiguous copy.

## Minimize allocations

Each element-wise operation (`+`, `-`, `*`, `/`) creates a new array.
If you are applying multiple operations, consider in-place updates when possible:

<!---FUN inplace_example-->

```kotlin
val x = mk.ndarray(mk[1.0, 2.0, 3.0])

x += 1.0
x *= 2.0
```

<!---END-->

This avoids intermediate arrays but mutates the original data.

## Be mindful of views

Indexing and slicing return views that share data with the base array.
Views are cheap to create, but repeated operations over many small views can be costly.
When performance matters, prefer working with a single contiguous array
and use `deepCopy()` to detach from views if needed.

## Reduce conversion overhead

Conversions allocate new memory:

- `toList`, `toArray`, `toIntArray`, and similar helpers
- `copy`, `deepCopy`, `flatten`

Avoid repeated conversions inside tight loops.
Instead, keep computations in ndarray form until the final output step.

## Numeric type choices

Using smaller types (`Float` vs `Double`, `Int` vs `Long`) reduces memory traffic.
Choose the smallest dtype that preserves the precision you need.

<seealso style="cards" title="Next steps">
<category ref="user-guide">
  <a href="engines-of-multik.md" summary="Understand DEFAULT, KOTLIN, and NATIVE engines."/>
  <a href="copies-and-views.md" summary="Learn when arrays share data and how to copy safely."/>
  <a href="shape-manipulation.md" summary="Reshape and reformat arrays without extra allocations."/>
</category>
</seealso>
