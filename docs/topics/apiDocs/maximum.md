# maximum

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for maximum() — returns an NDArray where each element is the larger of the
corresponding elements from two arrays.
</web-summary>

<card-summary>
Element-wise maximum of two arrays with the same shape.
</card-summary>

<link-summary>
maximum() — element-wise maximum of two NDArrays.
</link-summary>

Compares two arrays element by element and returns a new array where each element is the larger of
the two corresponding values. Both arrays must have the same shape.

## Signature

```kotlin
fun <T : Number, D : Dimension> MultiArray<T, D>.maximum(
    other: MultiArray<T, D>
): NDArray<T, D>
```

## Parameters

| Parameter | Type               | Description                                    |
|-----------|--------------------|------------------------------------------------|
| `other`   | `MultiArray<T, D>` | Array to compare against. Same shape required. |

**Returns:** `NDArray<T, D>` — each element is `max(this[i], other[i])`.

## Example

<!---FUN maximum_example-->

```kotlin
val a = mk.ndarray(mk[3, 1, 4])
val b = mk.ndarray(mk[1, 5, 2])

a.maximum(b)  // [3, 5, 4]
```

<!---END-->

## Pitfalls

* Complex types (`ComplexFloat`, `ComplexDouble`) throw `UnsupportedOperationException`.

<seealso style="cards">
<category ref="api-docs">
  <a href="minimum.md" summary="Element-wise minimum of two arrays."/>
  <a href="max.md" summary="Find the largest element in a single array."/>
  <a href="comparison-operations.md" summary="Overview of comparison operations."/>
</category>
</seealso>
