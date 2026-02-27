# Constants

<web-summary>
Multik does not define its own mathematical constants.
Use Kotlin's standard kotlin.math constants (PI, E) when working with NDArrays.
</web-summary>

<card-summary>
Mathematical constants for use with Multik — provided by kotlin.math.
</card-summary>

<link-summary>
Constants reference — use kotlin.math.PI, kotlin.math.E with Multik arrays.
</link-summary>

## Overview

Multik does not define its own mathematical constants.
Use the constants from Kotlin's standard library `kotlin.math`:

| Constant         | Value             | Type     |
|------------------|-------------------|----------|
| `kotlin.math.PI` | 3.141592653589793 | `Double` |
| `kotlin.math.E`  | 2.718281828459045 | `Double` |

## Usage with NDArrays

```kotlin
import kotlin.math.PI
import kotlin.math.E

val angles = mk.linspace<Double>(0, 1, 100) * PI  // 0 to PI
val exponential = mk.d1array(5) { E.pow(it.toDouble()) }
```

## Special values

`ComplexFloat` and `ComplexDouble` define these constants:

| Constant             | Type            | Value        |
|----------------------|-----------------|--------------|
| `ComplexFloat.zero`  | `ComplexFloat`  | `0.0 + 0.0i` |
| `ComplexFloat.one`   | `ComplexFloat`  | `1.0 + 0.0i` |
| `ComplexFloat.NaN`   | `ComplexFloat`  | `NaN + NaNi` |
| `ComplexDouble.zero` | `ComplexDouble` | `0.0 + 0.0i` |
| `ComplexDouble.one`  | `ComplexDouble` | `1.0 + 0.0i` |
| `ComplexDouble.NaN`  | `ComplexDouble` | `NaN + NaNi` |

<seealso style="cards">
<category ref="api-docs">
  <a href="scalars.md" summary="Supported numeric and complex element types."/>
  <a href="array-creation.md" summary="Functions for creating NDArrays."/>
</category>
</seealso>