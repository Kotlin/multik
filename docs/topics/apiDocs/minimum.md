# minimum

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for minimum() — returns an NDArray where each element is the smaller of the
corresponding elements from two arrays.
</web-summary>

<card-summary>
Element-wise minimum of two arrays with the same shape.
</card-summary>

<link-summary>
minimum() — element-wise minimum of two NDArrays.
</link-summary>

Compares two arrays element by element and returns a new array where each element is the smaller of
the two corresponding values. Both arrays must have the same shape.

## Signature

```kotlin
fun <T : Number, D : Dimension> MultiArray<T, D>.minimum(
    other: MultiArray<T, D>
): NDArray<T, D>
```

## Parameters

| Parameter | Type               | Description                                    |
|-----------|--------------------|------------------------------------------------|
| `other`   | `MultiArray<T, D>` | Array to compare against. Same shape required. |

**Returns:** `NDArray<T, D>` — each element is `min(this[i], other[i])`.

## Example

<!---FUN minimum_example-->

```kotlin
val a = mk.ndarray(mk[3, 1, 4])
val b = mk.ndarray(mk[1, 5, 2])

a.minimum(b)  // [1, 1, 2]
```

<!---END-->

## Pitfalls

* Complex types (`ComplexFloat`, `ComplexDouble`) throw `UnsupportedOperationException`.

<seealso style="cards">
<category ref="api-docs">
  <a href="maximum.md" summary="Element-wise maximum of two arrays."/>
  <a href="min.md" summary="Find the smallest element in a single array."/>
  <a href="comparison-operations.md" summary="Overview of comparison operations."/>
</category>
</seealso>
