# Array objects

<web-summary>
API reference for Multik array objects: NDArray class, scalar types, data types,
dimensions, indexing routines, iteration, and I/O operations.
</web-summary>

<card-summary>
Core array types and their properties, dimensions, indexing, iteration, and I/O.
</card-summary>

<link-summary>
Reference for NDArray, scalars, types, dimensions, indexing, iteration, and I/O.
</link-summary>

## Overview

This section documents the core array object model in Multik.

* **[](ndarray-class.md)** — the central class for multidimensional arrays, with properties, methods, and operator overloads.
* **[](scalars.md)** — supported element types: `Byte`, `Short`, `Int`, `Long`, `Float`, `Double`, `ComplexFloat`, `ComplexDouble`.
* **[](type.md)** — the `DataType` enum that identifies element types at runtime.
* **[](dimension.md)** — compile-time dimension markers `D1`–`D4` and `DN`.
* **[](indexing-routines.md)** — `Slice`, `RInt`, `ReadableView`, and the `slice`/`view` helpers.
* **[](iterating.md)** — `NDArrayIterator`, `MultiIndexProgression`, and collection-style operations.
* **[](IO-operations.md)** — file I/O (CSV, NPY, NPZ) on the JVM and in-memory conversions on all platforms.