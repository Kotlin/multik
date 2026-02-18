# identity

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.identity() — create a square identity matrix with ones on the diagonal.
</web-summary>

<card-summary>
Create an n-by-n identity matrix.
</card-summary>

<link-summary>
mk.identity() — square identity matrix.
</link-summary>

## Signatures

```kotlin
inline fun <reified T : Any> Multik.identity(n: Int): D2Array<T>

fun <T> Multik.identity(n: Int, dtype: DataType): D2Array<T>
```

## Parameters

| Parameter | Type       | Description                                     |
|-----------|------------|-------------------------------------------------|
| `n`       | `Int`      | Number of rows and columns.                     |
| `dtype`   | `DataType` | Element type (when not using reified generics). |

**Returns:** `D2Array<T>` — an `n × n` matrix with ones on the main diagonal and zeros elsewhere.

## Example

<!---FUN identity_example-->

```kotlin
val I = mk.identity<Double>(3)
// [[1.0, 0.0, 0.0],
//  [0.0, 1.0, 0.0],
//  [0.0, 0.0, 1.0]]

val intI = mk.identity<Int>(2, DataType.IntDataType)
// [[1, 0],
//  [0, 1]]
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="zeros.md" summary="Create zero-filled arrays."/>
  <a href="ones.md" summary="Create one-filled arrays."/>
  <a href="d2arrayIndices.md" summary="Create 2D arrays with custom diagonal logic."/>
</category>
</seealso>