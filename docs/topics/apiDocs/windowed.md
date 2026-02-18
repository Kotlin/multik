# windowed

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for windowed() — returns a 2D NDArray of sliding window segments over elements.
</web-summary>

<card-summary>
Sliding window over elements, returned as a 2D array.
</card-summary>

<link-summary>
windowed() — sliding window segments as a 2D array.
</link-summary>

## Signature

```kotlin
fun <T, D : Dimension> MultiArray<T, D>.windowed(
    size: Int,
    step: Int = 1,
    limit: Boolean = true
): NDArray<T, D2>
```

## Parameters

| Parameter | Type      | Description                                                                                                  |
|-----------|-----------|--------------------------------------------------------------------------------------------------------------|
| `size`    | `Int`     | Number of elements in each window.                                                                           |
| `step`    | `Int`     | Distance between the start of consecutive windows. Default: `1`.                                             |
| `limit`   | `Boolean` | If `true` (default), only full windows are included. If `false`, partial windows at the end are zero-padded. |

**Returns:** `NDArray<T, D2>` — a 2D array where each row is a window.

## Example

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])

a.windowed(3)
// [[1, 2, 3],
//  [2, 3, 4],
//  [3, 4, 5]]

a.windowed(3, step = 2)
// [[1, 2, 3],
//  [3, 4, 5]]
```

<seealso style="cards">
<category ref="api-docs">
  <a href="chunked.md" summary="Split into non-overlapping chunks."/>
  <a href="partition.md" summary="Split into matching and non-matching."/>
  <a href="drop.md" summary="Drop the first n elements."/>
</category>
</seealso>