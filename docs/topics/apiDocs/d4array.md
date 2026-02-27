# d4array

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.d4array() — create a 4D NDArray with a flat-index init lambda.
</web-summary>

<card-summary>
Create a 4D array where each element is computed by a flat-index init function.
</card-summary>

<link-summary>
mk.d4array() — 4D array from flat-index init lambda.
</link-summary>

Creates a 4D array of the given shape, where each element is computed by a flat-index `init`
function. Elements are filled in row-major order. For per-element `(i, j, k, m)` indices, use
[d4arrayIndices](d4arrayIndices.md).

## Signature

```kotlin
inline fun <reified T : Any> Multik.d4array(
    sizeD1: Int,
    sizeD2: Int,
    sizeD3: Int,
    sizeD4: Int,
    noinline init: (Int) -> T
): D4Array<T>
```

## Parameters

| Parameter | Type         | Description                                                                                                               |
|-----------|--------------|---------------------------------------------------------------------------------------------------------------------------|
| `sizeD1`  | `Int`        | Size of the first axis. Must be positive.                                                                                 |
| `sizeD2`  | `Int`        | Size of the second axis. Must be positive.                                                                                |
| `sizeD3`  | `Int`        | Size of the third axis. Must be positive.                                                                                 |
| `sizeD4`  | `Int`        | Size of the fourth axis. Must be positive.                                                                                |
| `init`    | `(Int) -> T` | Lambda that receives a flat index (0 to `sizeD1 * sizeD2 * sizeD3 * sizeD4 - 1`). Elements are filled in row-major order. |

**Returns:** `D4Array<T>`

## Example

<!---FUN d4array_example-->

```kotlin
val a = mk.d4array<Double>(2, 3, 4, 5) { it.toDouble() }
// shape (2, 3, 4, 5), 120 elements
```

<!---END-->

> For per-element `(i, j, k, m)` indices, use [d4arrayIndices](d4arrayIndices.md).
> {style="tip"}

<seealso style="cards">
<category ref="api-docs">
  <a href="d4arrayIndices.md" summary="Create 4D arrays with (i, j, k, m) init."/>
  <a href="d3array.md" summary="Create 3D arrays with a flat-index init lambda."/>
  <a href="dnarray.md" summary="Create N-D arrays with a flat-index init lambda."/>
</category>
</seealso>
