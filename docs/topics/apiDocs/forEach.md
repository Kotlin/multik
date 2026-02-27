# forEach

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for forEach() — performs an action on each element of an NDArray.
</web-summary>

<card-summary>
Perform an action on each element. Indexed variants provide element position.
</card-summary>

<link-summary>
forEach() — iterate over NDArray elements with an action.
</link-summary>

Performs the given action on each element of the array. Indexed variants provide the element's
flat index (`forEachIndexed`, 1D only) or multi-dimensional index (`forEachMultiIndexed`).

## Signatures

```kotlin
inline fun <T, D : Dimension> MultiArray<T, D>.forEach(
    action: (T) -> Unit
)

inline fun <T> MultiArray<T, D1>.forEachIndexed(
    action: (index: Int, T) -> Unit
)

inline fun <T, D : Dimension> MultiArray<T, D>.forEachMultiIndexed(
    action: (index: IntArray, T) -> Unit
)
```

## Parameters

| Parameter | Type                           | Description                              |
|-----------|--------------------------------|------------------------------------------|
| `action`  | `(T) -> Unit`                  | Action to perform on each element.       |
| `action`  | `(index: Int, T) -> Unit`      | Action with flat index (1D arrays only). |
| `action`  | `(index: IntArray, T) -> Unit` | Action with multi-dimensional index.     |

## Example

<!---FUN forEach_example-->

```kotlin
val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])

a.forEach { print("$it ") }
// 1 2 3 4

a.forEachMultiIndexed { idx, v ->
    println("${idx.toList()} -> $v")
}
// [0, 0] -> 1
// [0, 1] -> 2
// [1, 0] -> 3
// [1, 1] -> 4
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="onEach.md" summary="Perform an action and return the array."/>
  <a href="map.md" summary="Transform each element into a new array."/>
  <a href="asSequence.md" summary="Lazy sequence for chaining operations."/>
</category>
</seealso>
