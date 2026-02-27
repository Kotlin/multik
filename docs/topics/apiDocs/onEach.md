# onEach

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for onEach() — performs an action on each element and returns the array itself
for chaining.
</web-summary>

<card-summary>
Perform an action on each element and return the array for chaining.
</card-summary>

<link-summary>
onEach() — side-effect on each element, returns the array.
</link-summary>

Performs the given action on each element and returns the array itself, enabling method chaining.
Use it for side effects like logging or debugging within a pipeline of transformations.

## Signature

```kotlin
inline fun <T, D : Dimension, C : MultiArray<T, D>> C.onEach(
    action: (T) -> Unit
): C
```

## Parameters

| Parameter | Type          | Description                        |
|-----------|---------------|------------------------------------|
| `action`  | `(T) -> Unit` | Action to perform on each element. |

**Returns:** The original array `C`, enabling method chaining.

## Example

<!---FUN onEach_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3])

val result = a
    .onEach { print("$it ") }  // prints: 1 2 3
    .map { it * 2 }
// result: [2, 4, 6]
```

<!---END-->

> Unlike `forEach`, `onEach` returns the array itself. Use it for logging or debugging in a chain.
> {style="tip"}

<seealso style="cards">
<category ref="api-docs">
  <a href="forEach.md" summary="Iterate without returning the array."/>
  <a href="map.md" summary="Transform each element into a new array."/>
</category>
</seealso>
