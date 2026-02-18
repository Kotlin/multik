# d3array

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.d3array() — create a 3D NDArray with a flat-index init lambda.
</web-summary>

<card-summary>
Create a 3D array where each element is computed by a flat-index init function.
</card-summary>

<link-summary>
mk.d3array() — 3D array from flat-index init lambda.
</link-summary>

## Signature

```kotlin
inline fun <reified T : Any> Multik.d3array(
    sizeD1: Int,
    sizeD2: Int,
    sizeD3: Int,
    noinline init: (Int) -> T
): D3Array<T>
```

## Parameters

| Parameter | Type         | Description                                                                                                      |
|-----------|--------------|------------------------------------------------------------------------------------------------------------------|
| `sizeD1`  | `Int`        | Size of the first axis. Must be positive.                                                                        |
| `sizeD2`  | `Int`        | Size of the second axis. Must be positive.                                                                       |
| `sizeD3`  | `Int`        | Size of the third axis. Must be positive.                                                                        |
| `init`    | `(Int) -> T` | Lambda that receives a flat index (0 to `sizeD1 * sizeD2 * sizeD3 - 1`). Elements are filled in row-major order. |

**Returns:** `D3Array<T>`

## Example

<!---FUN d3array_example-->

```kotlin
val t = mk.d3array<Int>(2, 3, 4) { it }
// shape (2, 3, 4), elements 0..23
```

<!---END-->

> For per-element `(i, j, k)` indices, use [d3arrayIndices](d3arrayIndices.md).
> {style="tip"}

<seealso style="cards">
<category ref="api-docs">
  <a href="d3arrayIndices.md" summary="Create 3D arrays with (i, j, k) init."/>
  <a href="d2array.md" summary="Create 2D arrays with a flat-index init lambda."/>
  <a href="d4array.md" summary="Create 4D arrays with a flat-index init lambda."/>
</category>
</seealso>