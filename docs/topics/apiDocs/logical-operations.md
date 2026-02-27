# Logical operations

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for logical operations on Multik NDArrays — element-wise AND and OR
that return Int arrays with 1/0 values.
</web-summary>

<card-summary>
Element-wise logical AND and OR on NDArrays, returning Int arrays (1/0).
</card-summary>

<link-summary>
Logical operations — element-wise and, or returning Int arrays.
</link-summary>

## Overview

Multik provides two element-wise logical operations: `and` and `or`. Both are infix functions that
accept two arrays of the same shape and return an `NDArray<Int, D>` where each element is `1` (true)
or `0` (false).

## and

```kotlin
infix fun <T : Number, D : Dimension> MultiArray<T, D>.and(
    other: MultiArray<T, D>
): NDArray<Int, D>
```

Returns an `Int` array where each element is the logical AND of the corresponding elements.

### Truth rules

| Type                           | False value    | True value             |
|--------------------------------|----------------|------------------------|
| `Int`, `Long`, `Short`, `Byte` | `0`            | Non-zero (bitwise AND) |
| `Float`, `Double`              | `0.0` or `NaN` | Non-zero, non-NaN      |

### Example
{id="and-example"}

<!---FUN and_example-->

```kotlin
val a = mk.ndarray(mk[1, 0, 3, 0])
val b = mk.ndarray(mk[1, 1, 0, 0])

val c = a and b      // [1, 0, 0, 0]
```

<!---END-->

## or

```kotlin
infix fun <T : Number, D : Dimension> MultiArray<T, D>.or(
    other: MultiArray<T, D>
): NDArray<Int, D>
```

Returns an `Int` array where each element is the logical OR of the corresponding elements.

### Example
{id="or-example"}

<!---FUN or_example-->

```kotlin
val a = mk.ndarray(mk[1, 0, 3, 0])
val b = mk.ndarray(mk[1, 1, 0, 0])

val c = a or b       // [1, 1, 1, 0]
```

<!---END-->

## Pitfalls

* Both arrays must have the **same shape**.
* For integer types, bitwise AND/OR is used — `2 and 1` yields `0` (bitwise), not `1`.
* `Float`/`Double` `NaN` is treated as **false**.
* The return type is always `NDArray<Int, D>`, regardless of the input type.

<seealso style="cards">
<category ref="api-docs">
  <a href="arithmetic-operations.md" summary="Element-wise +, -, *, / operations."/>
  <a href="comparison-operations.md" summary="Element-wise minimum and maximum."/>
  <a href="all.md" summary="Check if all elements match a predicate."/>
  <a href="any.md" summary="Check if any element matches a predicate."/>
</category>
</seealso>
