# joinTo

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for joinTo() and joinToString() — builds a string representation of NDArray
elements with separators, prefix, and postfix.
</web-summary>

<card-summary>
Build a string from NDArray elements with separators, prefix, and postfix.
</card-summary>

<link-summary>
joinTo() / joinToString() — string representation of NDArray elements.
</link-summary>

Builds a string representation of the array's elements, with configurable separator, prefix,
postfix, element limit, and optional transform function. `joinTo` appends to an existing
`Appendable`; `joinToString` returns a `String` directly.

## Signatures

```kotlin
fun <T, D : Dimension, A : Appendable> MultiArray<T, D>.joinTo(
    buffer: A,
    separator: CharSequence = ", ",
    prefix: CharSequence = "",
    postfix: CharSequence = "",
    limit: Int = -1,
    truncated: CharSequence = "...",
    transform: ((T) -> CharSequence)? = null
): A

fun <T, D : Dimension> MultiArray<T, D>.joinToString(
    separator: CharSequence = ", ",
    prefix: CharSequence = "",
    postfix: CharSequence = "",
    limit: Int = -1,
    truncated: CharSequence = "...",
    transform: ((T) -> CharSequence)? = null
): String
```

## Parameters

| Parameter   | Type                         | Description                                       |
|-------------|------------------------------|---------------------------------------------------|
| `buffer`    | `A : Appendable`             | Target appendable (e.g., `StringBuilder`).        |
| `separator` | `CharSequence`               | String between elements. Default: `", "`.         |
| `prefix`    | `CharSequence`               | String before the first element. Default: `""`.   |
| `postfix`   | `CharSequence`               | String after the last element. Default: `""`.     |
| `limit`     | `Int`                        | Max elements to include. `-1` for all.            |
| `truncated` | `CharSequence`               | Shown when limit is exceeded. Default: `"..."`.   |
| `transform` | `((T) -> CharSequence)?`     | Custom element formatting. Default: `toString()`. |

**Returns:** `joinTo` returns the `buffer`; `joinToString` returns a `String`.

## Example

<!---FUN joinToString_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])

a.joinToString()                           // "1, 2, 3, 4, 5"
a.joinToString(separator = " | ")          // "1 | 2 | 3 | 4 | 5"
a.joinToString(prefix = "[", postfix = "]") // "[1, 2, 3, 4, 5]"
a.joinToString(limit = 3)                  // "1, 2, 3, ..."
a.joinToString { (it * 10).toString() }    // "10, 20, 30, 40, 50"
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="toList.md" summary="Convert to a List."/>
  <a href="forEach.md" summary="Iterate over elements."/>
  <a href="map.md" summary="Transform each element."/>
</category>
</seealso>
