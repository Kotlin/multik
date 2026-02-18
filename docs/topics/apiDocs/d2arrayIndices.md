# d2arrayIndices

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.d2arrayIndices() — create a 2D NDArray with an (i, j) index-based init lambda.
</web-summary>

<card-summary>
Create a 2D array where each element is computed from its row and column indices.
</card-summary>

<link-summary>
mk.d2arrayIndices() — 2D array from (i, j) init lambda.
</link-summary>

## Signature

```kotlin
inline fun <reified T : Any> Multik.d2arrayIndices(
    sizeD1: Int,
    sizeD2: Int,
    init: (i: Int, j: Int) -> T
): D2Array<T>
```

## Parameters

| Parameter | Type                    | Description                                              |
|-----------|-------------------------|----------------------------------------------------------|
| `sizeD1`  | `Int`                   | Number of rows. Must be positive.                        |
| `sizeD2`  | `Int`                   | Number of columns. Must be positive.                     |
| `init`    | `(i: Int, j: Int) -> T` | Lambda that receives row index `i` and column index `j`. |

**Returns:** `D2Array<T>`

## Example

<!---FUN d2arrayIndices_example-->

```kotlin
val m = mk.d2arrayIndices<Int>(3, 3) { i, j -> i + j }
// [[0, 1, 2],
//  [1, 2, 3],
//  [2, 3, 4]]

val identity = mk.d2arrayIndices<Double>(4, 4) { i, j ->
    if (i == j) 1.0 else 0.0
}
```

<!---END-->

> For simple cases where you do not need per-axis indices, [d2array](d2array.md) with a flat index is slightly faster.
> {style="tip"}

<seealso style="cards">
<category ref="api-docs">
  <a href="d2array.md" summary="Create 2D arrays with a flat-index init lambda."/>
  <a href="d3arrayIndices.md" summary="Create 3D arrays with (i, j, k) init."/>
  <a href="identity.md" summary="Create identity matrices directly."/>
</category>
</seealso>