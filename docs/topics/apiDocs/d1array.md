# d1array

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.d1array() — create a 1D NDArray with an init lambda.
</web-summary>

<card-summary>
Create a 1D array where each element is computed by an init function.
</card-summary>

<link-summary>
mk.d1array() — 1D array from init lambda.
</link-summary>

Creates a 1D array of the given size, where each element is computed by the `init` function
that receives the element's index.

## Signature

```kotlin
inline fun <reified T : Any> Multik.d1array(
    sizeD1: Int,
    noinline init: (Int) -> T
): D1Array<T>
```

## Parameters

| Parameter | Type         | Description                                                                       |
|-----------|--------------|-----------------------------------------------------------------------------------|
| `sizeD1`  | `Int`        | Number of elements. Must be positive.                                             |
| `init`    | `(Int) -> T` | Lambda that receives the index (0 to `sizeD1 - 1`) and returns the element value. |

**Returns:** `D1Array<T>`

## Example

<!---FUN d1array_example-->

```kotlin
val a = mk.d1array<Int>(5) { it * 2 }
// [0, 2, 4, 6, 8]

val b = mk.d1array<Double>(3) { it + 0.5 }
// [0.5, 1.5, 2.5]
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="d2array.md" summary="Create 2D arrays with a flat-index init lambda."/>
  <a href="ndarray.md" summary="Create arrays from lists and collections."/>
  <a href="zeros.md" summary="Create zero-filled arrays."/>
</category>
</seealso>
