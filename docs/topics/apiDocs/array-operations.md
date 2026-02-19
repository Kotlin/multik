# Array operations

<web-summary>
API reference for NDArray operations in Multik — arithmetic, logical, comparison,
and universal collection-style operations.
</web-summary>

<card-summary>
Arithmetic, logical, comparison, and collection-style operations on NDArrays.
</card-summary>

<link-summary>
Array operations — arithmetic, logical, comparison, and universal operations.
</link-summary>

## Overview

Multik NDArrays support a rich set of operations organized into four categories:

### Arithmetic

| Operation      | Operator / Function | Description                  |
|----------------|---------------------|------------------------------|
| Addition       | `+`, `+=`           | Element-wise addition.       |
| Subtraction    | `-`, `-=`           | Element-wise subtraction.    |
| Multiplication | `*`, `*=`           | Element-wise multiplication. |
| Division       | `/`, `/=`           | Element-wise division.       |
| Unary negation | `-a`                | Negate all elements.         |

See [](arithmetic-operations.md) for details.

### Logical

| Operation   | Function | Description                      |
|-------------|----------|----------------------------------|
| Logical AND | `and`    | Element-wise AND, returns `Int`. |
| Logical OR  | `or`     | Element-wise OR, returns `Int`.  |

See [](logical-operations.md) for details.

### Comparison

| Operation        | Function  | Description                         |
|------------------|-----------|-------------------------------------|
| Element-wise min | `minimum` | Element-wise minimum of two arrays. |
| Element-wise max | `maximum` | Element-wise maximum of two arrays. |

See [](comparison-operations.md) for details.

### Universal (collection-style)

Functions like `map`, `filter`, `fold`, `reduce`, `forEach`, `sorted`, and many more.
These follow Kotlin collection conventions and work on any `MultiArray<T, D>`.

See [](universal-operations.md) for the full list.

<seealso style="cards">
<category ref="api-docs">
  <a href="arithmetic-operations.md" summary="Addition, subtraction, multiplication, division."/>
  <a href="logical-operations.md" summary="Element-wise and, or operations."/>
  <a href="comparison-operations.md" summary="Element-wise minimum and maximum."/>
  <a href="universal-operations.md" summary="Collection-style operations — map, filter, fold, etc."/>
</category>
</seealso>