# average

<!---IMPORT samples.docs.apiDocs.ArrayOperations-->

<web-summary>
API reference for average() — returns the arithmetic mean of all elements in an NDArray as a Double.
</web-summary>

<card-summary>
Returns the arithmetic mean of all elements as a Double.
</card-summary>

<link-summary>
average() — compute the mean of NDArray elements.
</link-summary>

## Signature

```kotlin
fun <T : Number, D : Dimension> MultiArray<T, D>.average(): Double
```

**Returns:** The arithmetic mean of all elements as `Double`.

## Example

<!---FUN average_example-->

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5])
a.average()  // 3.0

val b = mk.ndarray(mk[1.5, 2.5])
b.average()  // 2.0
```

<!---END-->

<seealso style="cards">
<category ref="api-docs">
  <a href="sum.md" summary="Sum of all elements."/>
  <a href="count.md" summary="Count elements."/>
  <a href="max.md" summary="Find the largest element."/>
  <a href="min.md" summary="Find the smallest element."/>
</category>
</seealso>