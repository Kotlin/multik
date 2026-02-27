# clip

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for clip() — clamps every element to a given range, returning a new array.
</web-summary>

<card-summary>
Clamp array elements to a min..max range.
</card-summary>

<link-summary>
clip() — clamp NDArray elements to a range.
</link-summary>

Returns a new array where every element is clamped to the `[min, max]` range. Elements below
`min` are set to `min`; elements above `max` are set to `max`.

## Signature

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.clip(
    min: T, max: T
): NDArray<T, D> where T : Comparable<T>, T : Number
```

## Parameters

| Parameter | Type | Description                               |
|-----------|------|-------------------------------------------|
| `min`     | `T`  | Lower bound. Elements below are set here. |
| `max`     | `T`  | Upper bound. Elements above are set here. |

**Returns:** `NDArray<T, D>` — a new array with the same shape, all elements in `min..max`.

## Example

<!---FUN clip_example-->

```kotlin
val a = mk.ndarray(mk[1, 5, 10, 15, 20])
a.clip(5, 15) // [5, 5, 10, 15, 15]

val m = mk.ndarray(mk[mk[0.0, 0.5], mk[1.0, 1.5]])
m.clip(0.2, 1.0)
// [[0.2, 0.5],
//  [1.0, 1.0]]
```

<!---END-->

> `clip` requires `min <= max`. An `IllegalArgumentException` is thrown otherwise.
> {style="warning"}

<seealso style="cards">
<category ref="api-docs">
  <a href="minimum.md" summary="Element-wise minimum of two arrays."/>
  <a href="maximum.md" summary="Element-wise maximum of two arrays."/>
  <a href="map.md" summary="Transform each element with a custom function."/>
</category>
</seealso>
