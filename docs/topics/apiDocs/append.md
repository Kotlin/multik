# append

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for append() — appends elements or arrays, returning a new array.
</web-summary>

<card-summary>
Append elements or another array, returning a new array.
</card-summary>

<link-summary>
append() — append elements or arrays to an NDArray.
</link-summary>

Appends individual elements or another array, returning a new array. The first two overloads
flatten both arrays to 1D before concatenating; the axis-based overload preserves dimensionality.

## Signatures

```kotlin
// Append individual elements (flattens to 1D)
fun <T, D : Dimension> MultiArray<T, D>.append(vararg value: T): D1Array<T>

// Append another array (flattens to 1D)
infix fun <T, D : Dimension, ID : Dimension> MultiArray<T, D>.append(
    arr: MultiArray<T, ID>
): D1Array<T>

// Append along a specific axis (preserves dimensionality)
fun <T, D : Dimension> MultiArray<T, D>.append(
    arr: MultiArray<T, D>, axis: Int
): NDArray<T, D>
```

## Parameters

| Parameter | Type              | Description                                                        |
|-----------|-------------------|--------------------------------------------------------------------|
| `value`   | `vararg T`        | Individual elements to append.                                     |
| `arr`     | `MultiArray<T, *>` | Array whose elements are appended.                                |
| `axis`    | `Int`             | Axis along which to concatenate (only for the shaped overload).    |

**Returns:**
- Without `axis`: `D1Array<T>` — a new flat 1D array with the appended elements.
- With `axis`: `NDArray<T, D>` — a new array concatenated along the given axis.

## Example

<!---FUN append_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])

// Append individual elements
a.append(4, 5)               // [1, 2, 3, 4, 5]

// Append another array (flattened)
val b = mk.ndarray(mk[10, 20])
a append b                   // [1, 2, 3, 10, 20]

// Append along an axis
val m1 = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
val m2 = mk.ndarray(mk[mk[5, 6]])
m1.append(m2, axis = 0)
// [[1, 2],
//  [3, 4],
//  [5, 6]]
```

<!---END-->

> The first two overloads always **flatten** both arrays to 1D before concatenating.
> Use the `axis` overload or [cat](ndarray-class.md) to preserve dimensionality.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="chunked.md" summary="Split into fixed-size chunks."/>
  <a href="drop.md" summary="Drop the first n elements."/>
</category>
</seealso>
