# Statistics

<web-summary>
API reference for statistical functions in Multik — descriptive statistics (mean, median, average)
via mk.stat.
</web-summary>

<card-summary>
Statistical functions — mean, median, weighted average via mk.stat.
</card-summary>

<link-summary>
Statistics — descriptive statistics via mk.stat.
</link-summary>

## Overview

The `mk.stat` entry point provides statistical operations on ndarrays.
Functions are defined in the `Statistics` interface.

## Available functions

| Function                                       | Description                                        |
|------------------------------------------------|----------------------------------------------------|
| [`mean`](descriptive-statistics.md#mean)       | Arithmetic mean of all elements, or along an axis. |
| [`median`](descriptive-statistics.md#median)   | Median value of all elements.                      |
| [`average`](descriptive-statistics.md#average) | Weighted average of all elements.                  |

For details and examples, see [Descriptive Statistics](descriptive-statistics.md).

## Quick example

```kotlin
val a = mk.ndarray(mk[1.0, 2.0, 3.0, 4.0, 5.0])

mk.stat.mean(a)      // 3.0
mk.stat.median(a)    // 3.0
mk.stat.average(a)   // 3.0 (uniform weights)
```

<seealso style="cards">
<category ref="api-docs">
  <a href="descriptive-statistics.md" summary="mean, median, and weighted average in detail."/>
  <a href="mathematical.md" summary="Mathematical functions via mk.math."/>
  <a href="linear-algebra.md" summary="Linear algebra operations via mk.linalg."/>
</category>
</seealso>