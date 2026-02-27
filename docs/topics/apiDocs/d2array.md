# d2array

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.d2array() — create a 2D NDArray with a flat-index init lambda.
</web-summary>

<card-summary>
Create a 2D array where each element is computed by a flat-index init function.
</card-summary>

<link-summary>
mk.d2array() — 2D array from flat-index init lambda.
</link-summary>

Creates a 2D array of the given shape, where each element is computed by a flat-index `init`
function. Elements are filled in row-major order. For per-element `(i, j)` indices, use
[d2arrayIndices](d2arrayIndices.md).

## Signature

```kotlin
inline fun <reified T : Any> Multik.d2array(
    sizeD1: Int,
    sizeD2: Int,
    noinline init: (Int) -> T
): D2Array<T>
```

## Parameters

| Parameter | Type         | Description                                                                                                                           |
|-----------|--------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `sizeD1`  | `Int`        | Number of rows. Must be positive.                                                                                                     |
| `sizeD2`  | `Int`        | Number of columns. Must be positive.                                                                                                  |
| `init`    | `(Int) -> T` | Lambda that receives a flat index (0 to `sizeD1 * sizeD2 - 1`) and returns the element value. Elements are filled in row-major order. |

**Returns:** `D2Array<T>`

## Example

<!---FUN d2array_example-->

```kotlin
val m = mk.d2array<Int>(2, 3) { it }
// [[0, 1, 2],
//  [3, 4, 5]]

val identity = mk.d2array<Double>(3, 3) { if (it / 3 == it % 3) 1.0 else 0.0 }
```

<!---END-->

> The init lambda receives a **flat** index. For per-element `(i, j)` indices, use [d2arrayIndices](d2arrayIndices.md).
> {style="tip"}

<seealso style="cards">
<category ref="api-docs">
  <a href="d2arrayIndices.md" summary="Create 2D arrays with (i, j) index-based init."/>
  <a href="d1array.md" summary="Create 1D arrays with an init lambda."/>
  <a href="d3array.md" summary="Create 3D arrays with a flat-index init lambda."/>
</category>
</seealso>
