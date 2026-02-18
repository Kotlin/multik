# d3arrayIndices

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.d3arrayIndices() — create a 3D NDArray with an (i, j, k) index-based init lambda.
</web-summary>

<card-summary>
Create a 3D array where each element is computed from its three axis indices.
</card-summary>

<link-summary>
mk.d3arrayIndices() — 3D array from (i, j, k) init lambda.
</link-summary>

## Signature

```kotlin
inline fun <reified T : Any> Multik.d3arrayIndices(
    sizeD1: Int,
    sizeD2: Int,
    sizeD3: Int,
    init: (i: Int, j: Int, k: Int) -> T
): D3Array<T>
```

## Parameters

| Parameter | Type                            | Description                                   |
|-----------|---------------------------------|-----------------------------------------------|
| `sizeD1`  | `Int`                           | Size of the first axis. Must be positive.     |
| `sizeD2`  | `Int`                           | Size of the second axis. Must be positive.    |
| `sizeD3`  | `Int`                           | Size of the third axis. Must be positive.     |
| `init`    | `(i: Int, j: Int, k: Int) -> T` | Lambda that receives indices along each axis. |

**Returns:** `D3Array<T>`

## Example

<!---FUN d3arrayIndices_example-->

```kotlin
val t = mk.d3arrayIndices<Int>(2, 3, 4) { i, j, k -> i * 100 + j * 10 + k }
// t[1, 2, 3] == 123
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="d3array.md" summary="Create 3D arrays with a flat-index init lambda."/>
  <a href="d2arrayIndices.md" summary="Create 2D arrays with (i, j) init."/>
  <a href="d4arrayIndices.md" summary="Create 4D arrays with (i, j, k, m) init."/>
</category>
</seealso>