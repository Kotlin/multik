# associate

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for associate() — builds a Map from NDArray elements using a transform function
that produces key-value pairs.
</web-summary>

<card-summary>
Build a Map from NDArray elements using a key-value transform function.
</card-summary>

<link-summary>
associate() — create a Map from NDArray elements.
</link-summary>

## Signatures

```kotlin
inline fun <T, D : Dimension, K, V> MultiArray<T, D>.associate(
    transform: (T) -> Pair<K, V>
): Map<K, V>

inline fun <T, D : Dimension, K> MultiArray<T, D>.associateBy(
    keySelector: (T) -> K
): Map<K, T>

inline fun <T, D : Dimension, K, V> MultiArray<T, D>.associateBy(
    keySelector: (T) -> K,
    valueTransform: (T) -> V
): Map<K, V>

inline fun <K, D : Dimension, V> MultiArray<K, D>.associateWith(
    valueSelector: (K) -> V
): Map<K, V>
```

## Parameters

| Parameter        | Type                | Description                                      |
|------------------|---------------------|--------------------------------------------------|
| `transform`      | `(T) -> Pair<K, V>` | Produces a key-value pair from each element.     |
| `keySelector`    | `(T) -> K`          | Extracts the key from each element.              |
| `valueTransform` | `(T) -> V`          | Transforms the element into the map value.       |
| `valueSelector`  | `(K) -> V`          | Produces the value for each element used as key. |

**Returns:** `Map<K, V>` preserving iteration order. Duplicate keys keep the last value.

## Example

<!---FUN associate_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])

a.associate { it to it * it }       // {1=1, 2=4, 3=9}
a.associateBy { it * 10 }           // {10=1, 20=2, 30=3}
a.associateWith { it.toString() }   // {1="1", 2="2", 3="3"}
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="groupNDArray.md" summary="Group elements by key into NDArrays."/>
  <a href="map.md" summary="Transform each element."/>
  <a href="toList.md" summary="Convert to a List."/>
</category>
</seealso>