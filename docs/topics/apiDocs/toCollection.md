# toCollection

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for toCollection(), toSet(), toHashSet(), and toMutableSet() — append NDArray
elements to Kotlin collections.
</web-summary>

<card-summary>
Append NDArray elements to a mutable collection, or convert to Set/HashSet.
</card-summary>

<link-summary>
toCollection() — convert NDArray elements to Kotlin collections.
</link-summary>

## Signatures

```kotlin
fun <T, D : Dimension, C : MutableCollection<in T>> MultiArray<T, D>.toCollection(
    destination: C
): C

fun <T, D : Dimension> MultiArray<T, D>.toSet(): Set<T>
fun <T, D : Dimension> MultiArray<T, D>.toHashSet(): HashSet<T>
fun <T, D : Dimension> MultiArray<T, D>.toMutableSet(): MutableSet<T>
```

## Parameters

| Parameter     | Type                          | Description                              |
|---------------|-------------------------------|------------------------------------------|
| `destination` | `C : MutableCollection<in T>` | Target collection to append elements to. |

**Returns:**

- `toCollection`: the `destination` with all elements added.
- `toSet` / `toHashSet` / `toMutableSet`: a new set containing all unique elements.

## Example

<!---FUN toCollection_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 2, 3, 3, 3])

a.toSet()                            // {1, 2, 3}
a.toHashSet()                        // HashSet {1, 2, 3}

val list = mutableListOf(0)
a.toCollection(list)                 // [0, 1, 2, 2, 3, 3, 3]
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="toList.md" summary="Convert to a List."/>
  <a href="toPrimitiveArray.md" summary="Convert to a primitive array."/>
  <a href="distinct.md" summary="Get unique elements as an NDArray."/>
</category>
</seealso>