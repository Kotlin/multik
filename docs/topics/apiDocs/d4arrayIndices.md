# d4arrayIndices

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.d4arrayIndices() — create a 4D NDArray with an (i, j, k, m) index-based init lambda.
</web-summary>

<card-summary>
Create a 4D array where each element is computed from its four axis indices.
</card-summary>

<link-summary>
mk.d4arrayIndices() — 4D array from (i, j, k, m) init lambda.
</link-summary>

Creates a 4D array of the given shape, where each element is computed from its four axis indices
`(i, j, k, m)`. Useful when the element value depends on its position in the 4D space.

## Signature

```kotlin
inline fun <reified T : Any> Multik.d4arrayIndices(
    sizeD1: Int,
    sizeD2: Int,
    sizeD3: Int,
    sizeD4: Int,
    init: (i: Int, j: Int, k: Int, m: Int) -> T
): D4Array<T>
```

## Parameters

| Parameter | Type                                    | Description                                               |
|-----------|-----------------------------------------|-----------------------------------------------------------|
| `sizeD1`  | `Int`                                   | Size of the first axis. Must be positive.                 |
| `sizeD2`  | `Int`                                   | Size of the second axis. Must be positive.                |
| `sizeD3`  | `Int`                                   | Size of the third axis. Must be positive.                 |
| `sizeD4`  | `Int`                                   | Size of the fourth axis. Must be positive.                |
| `init`    | `(i: Int, j: Int, k: Int, m: Int) -> T` | Lambda that receives indices along each of the four axes. |

**Returns:** `D4Array<T>`

## Example

<!---FUN d4arrayIndices_example-->

```kotlin
val a = mk.d4arrayIndices<Int>(2, 2, 2, 2) { i, j, k, m ->
    i * 8 + j * 4 + k * 2 + m
}
// a[1, 1, 1, 1] == 15
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="d4array.md" summary="Create 4D arrays with a flat-index init lambda."/>
  <a href="d3arrayIndices.md" summary="Create 3D arrays with (i, j, k) init."/>
  <a href="dnarray.md" summary="Create N-D arrays with a flat-index init lambda."/>
</category>
</seealso>
