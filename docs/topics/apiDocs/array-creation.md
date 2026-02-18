# Array creation

<web-summary>
API reference for all Multik array creation functions: ndarray, ndarrayOf, d1array–d4array,
dnarray, zeros, ones, identity, arange, linspace, rand, meshgrid, and toNDArray.
</web-summary>

<card-summary>
All functions for creating NDArrays — from literals, init lambdas, ranges, and collections.
</card-summary>

<link-summary>
Array creation reference — ndarray, zeros, ones, arange, linspace, rand, and more.
</link-summary>

## Overview

All creation functions are called on the `Multik` object (aliased as `mk`).
They are grouped by usage pattern:

### From data

| Function         | Description                                                                                  |
|------------------|----------------------------------------------------------------------------------------------|
| [](ndarray.md)   | Create from nested lists, collections, or primitive arrays with `mk[]` syntax.               |
| [](ndarrayOf.md) | Create a 1D array from varargs.                                                              |
| [](toNDArray.md) | Extension function to convert `Iterable`, `List<List<T>>`, or `Array<*Array>` to an NDArray. |

### With init lambda

| Function              | Description                                           |
|-----------------------|-------------------------------------------------------|
| [](d1array.md)        | 1D array from `(Int) -> T` init function.             |
| [](d2array.md)        | 2D array from `(Int) -> T` flat-index init function.  |
| [](d2arrayIndices.md) | 2D array from `(i, j) -> T` init function.            |
| [](d3array.md)        | 3D array from `(Int) -> T` flat-index init function.  |
| [](d3arrayIndices.md) | 3D array from `(i, j, k) -> T` init function.         |
| [](d4array.md)        | 4D array from `(Int) -> T` flat-index init function.  |
| [](d4arrayIndices.md) | 4D array from `(i, j, k, m) -> T` init function.      |
| [](dnarray.md)        | N-D array from `(Int) -> T` flat-index init function. |

### Filled arrays

| Function        | Description                                |
|-----------------|--------------------------------------------|
| [](zeros.md)    | Array filled with zeros.                   |
| [](ones.md)     | Array filled with ones.                    |
| [](identity.md) | Square identity matrix (ones on diagonal). |

### Ranges and sequences

| Function        | Description                                                       |
|-----------------|-------------------------------------------------------------------|
| [](arange.md)   | Evenly spaced values within a half-open interval `[start, stop)`. |
| [](linspace.md) | Evenly spaced values within a closed interval `[start, stop]`.    |

### Random

| Function    | Description                      |
|-------------|----------------------------------|
| [](rand.md) | Array filled with random values. |

### Grids

| Function        | Description                                         |
|-----------------|-----------------------------------------------------|
| [](meshgrid.md) | Coordinate matrices from two 1D coordinate vectors. |