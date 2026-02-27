# meshgrid

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.meshgrid() — create 2D coordinate matrices from two 1D coordinate vectors.
</web-summary>

<card-summary>
Create coordinate matrices from two 1D vectors for grid-based computation.
</card-summary>

<link-summary>
mk.meshgrid() — 2D coordinate grids from 1D vectors.
</link-summary>

Creates a pair of 2D coordinate matrices from two 1D coordinate vectors. Useful for evaluating
functions on a grid or creating plotting meshes.

## Signature

```kotlin
fun <T : Number> Multik.meshgrid(
    x: MultiArray<T, D1>,
    y: MultiArray<T, D1>
): Pair<D2Array<T>, D2Array<T>>
```

## Parameters

| Parameter | Type                | Description                |
|-----------|---------------------|----------------------------|
| `x`       | `MultiArray<T, D1>` | 1D array of x-coordinates. |
| `y`       | `MultiArray<T, D1>` | 1D array of y-coordinates. |

**Returns:** `Pair<D2Array<T>, D2Array<T>>` — two matrices `(X, Y)` where:

* `X` has shape `(y.size, x.size)` — each row is a copy of `x`.
* `Y` has shape `(y.size, x.size)` — each column is a copy of `y`.

## Example

<!---FUN meshgrid_example-->

```kotlin
val x = mk.ndarray(mk[1, 2, 3])
val y = mk.ndarray(mk[4, 5])

val (X, Y) = mk.meshgrid(x, y)

// X:
// [[1, 2, 3],
//  [1, 2, 3]]

// Y:
// [[4, 4, 4],
//  [5, 5, 5]]
```

<!---END-->

This is useful for evaluating functions on a grid:

<!---FUN meshgrid_grid_function_example-->

```kotlin
val x = mk.linspace<Double>(0.0, 1.0, 10)
val y = mk.linspace<Double>(0.0, 1.0, 10)
val (X, Y) = mk.meshgrid(x, y)
// Compute f(x, y) = x^2 + y^2 element-wise on the grid
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="linspace.md" summary="Create evenly spaced 1D arrays for grid coordinates."/>
  <a href="arange.md" summary="Create arrays with a fixed step."/>
</category>
</seealso>
