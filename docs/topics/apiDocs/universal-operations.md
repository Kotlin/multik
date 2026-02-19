# Universal operations

<web-summary>
API reference for collection-style universal operations on Multik NDArrays — map, filter,
fold, reduce, forEach, sorted, and more.
</web-summary>

<card-summary>
Collection-style operations on NDArrays — map, filter, fold, reduce, sorted, and more.
</card-summary>

<link-summary>
Universal operations — collection-style functions for NDArrays.
</link-summary>

## Overview

Universal operations are extension functions on `MultiArray<T, D>` that follow Kotlin collection
conventions. They provide familiar APIs for transforming, filtering, aggregating, and converting arrays.

### Predicates

| Function        | Description                                   |
|-----------------|-----------------------------------------------|
| [](all.md)      | `true` if all elements match a predicate.     |
| [](any.md)      | `true` if at least one element matches.       |
| [](contains.md) | `true` if the array contains a given element. |

### Traversal

| Function       | Description                                           |
|----------------|-------------------------------------------------------|
| [](forEach.md) | Apply an action to each element.                      |
| [](onEach.md)  | Apply an action to each element and return the array. |

### Transformation

| Function       | Description                                              |
|----------------|----------------------------------------------------------|
| [](map.md)     | Transform each element, preserving shape.                |
| [](flatMap.md) | Transform each element into an iterable, flatten to 1D.  |
| [](scan.md)    | Running accumulation, returning all intermediate values. |

### Filtering

| Function        | Description                                    |
|-----------------|------------------------------------------------|
| [](filter.md)   | Elements matching a predicate, returned as 1D. |
| [](find.md)     | First element matching a predicate, or `null`. |
| [](distinct.md) | Unique elements as a 1D array.                 |

### Aggregation

| Function       | Description                                      |
|----------------|--------------------------------------------------|
| [](fold.md)    | Accumulate from an initial value.                |
| [](reduce.md)  | Accumulate without an initial value.             |
| [](sum.md)     | Sum of all elements.                             |
| [](average.md) | Average of all elements.                         |
| [](count.md)   | Count elements, optionally matching a predicate. |
| [](max.md)     | Largest element, or `null` if empty.             |
| [](min.md)     | Smallest element, or `null` if empty.            |
| [](maximum.md) | Element-wise maximum of two arrays.              |
| [](minimum.md) | Element-wise minimum of two arrays.              |

### Search

| Function       | Description                                   |
|----------------|-----------------------------------------------|
| [](first.md)   | First element, or first matching a predicate. |
| [](last.md)    | Last element, or last matching a predicate.   |
| [](indexOf.md) | Flat index of the first occurrence, or `-1`.  |

### Ordering

| Function        | Description                               |
|-----------------|-------------------------------------------|
| [](sorted.md)   | New array with elements sorted.           |
| [](reversed.md) | New array with elements in reverse order. |

### Partitioning

| Function         | Description                                 |
|------------------|---------------------------------------------|
| [](partition.md) | Split into (matching, non-matching) pair.   |
| [](chunked.md)   | Split into fixed-size chunks as a 2D array. |
| [](windowed.md)  | Sliding window as a 2D array.               |
| [](drop.md)      | Drop the first *n* elements (1D only).      |

### Grouping

| Function            | Description                                    |
|---------------------|------------------------------------------------|
| [](groupNDArray.md) | Group elements by key into a map of 1D arrays. |
| [](associate.md)    | Build a map from key-value pairs.              |
| [](intersect.md)    | Set intersection with another collection.      |

### Conversion

| Function                | Description                                    |
|-------------------------|------------------------------------------------|
| [](toList.md)           | Flat `List<T>` of all elements.                |
| [](toCollection.md)     | Append all elements to a mutable collection.   |
| [](toPrimitiveArray.md) | Convert to a Kotlin primitive array.           |
| [](toArray.md)          | Convert to nested Kotlin arrays.               |
| [](asSequence.md)       | Lazy `Sequence<T>` over elements.              |
| [](toType.md)           | Convert element type (e.g. `Int` to `Double`). |
| [](joinTo.md)           | Build a string representation.                 |

<seealso style="cards">
<category ref="api-docs">
  <a href="arithmetic-operations.md" summary="Element-wise +, -, *, / operations."/>
  <a href="logical-operations.md" summary="Element-wise and, or operations."/>
  <a href="iterating.md" summary="NDArrayIterator and MultiIndexProgression."/>
</category>
</seealso>