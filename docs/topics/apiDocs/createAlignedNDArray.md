# createAlignedNDArray

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.createAlignedNDArray() — create an NDArray from ragged nested collections with automatic padding.
</web-summary>

<card-summary>
Create an NDArray from ragged (jagged) nested lists or arrays, padding shorter sequences to uniform length.
</card-summary>

<link-summary>
mk.createAlignedNDArray() — NDArray from ragged collections with padding.
</link-summary>

> This is an experimental API. It may be changed or removed in future releases.
> Requires `@OptIn(ExperimentalMultikApi::class)`.
> {style="warning"}

## Signature

```kotlin
inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: List<List<T>>, filling: Double = 0.0
): D2Array<T>

inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: List<List<List<T>>>, filling: Double = 0.0
): D3Array<T>

inline fun <reified T : Number> Multik.createAlignedNDArray(
    data: List<List<List<List<T>>>>, filling: Double = 0.0
): D4Array<T>
```

Each variant also has an `Array` overload that delegates to the `List` version.

## Parameters

| Parameter | Type             | Description                                                                                                            |
|-----------|------------------|------------------------------------------------------------------------------------------------------------------------|
| `data`    | nested List or Array | Ragged nested collection. Inner sequences may have different lengths.                                                  |
| `filling` | `Double`         | Value used to pad shorter sequences to the maximum length at each depth. Converted to `T`. Defaults to `0.0`. |

**Returns:** `D2Array<T>`, `D3Array<T>`, or `D4Array<T>` depending on nesting depth.

**Throws:** `IllegalArgumentException` if `data` is empty.

## Example

<!---FUN createAlignedNDArray_example-->

```kotlin
val ragged = listOf(
    listOf(1, 2, 3),
    listOf(4, 5),
    listOf(6)
)
val aligned = mk.createAlignedNDArray(ragged)
// [[1, 2, 3],
//  [4, 5, 0],
//  [6, 0, 0]]

val padded = mk.createAlignedNDArray(ragged, filling = -1.0)
// [[1,  2,  3],
//  [4,  5, -1],
//  [6, -1, -1]]
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="ndarray.md" summary="Create arrays from flat data with explicit shape."/>
  <a href="d2array.md" summary="Create 2D arrays with an init lambda."/>
  <a href="toNDArray.md" summary="Convert standard collections to NDArray."/>
</category>
</seealso>
