# ndarrayOf

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.ndarrayOf() — create a 1D NDArray from varargs.
</web-summary>

<card-summary>
Create a 1D array from vararg elements.
</card-summary>

<link-summary>
mk.ndarrayOf() — 1D array from varargs.
</link-summary>

## Signature

```kotlin
fun Multik.ndarrayOf(vararg items: Byte): D1Array<Byte>
fun Multik.ndarrayOf(vararg items: Short): D1Array<Short>
fun Multik.ndarrayOf(vararg items: Int): D1Array<Int>
fun Multik.ndarrayOf(vararg items: Long): D1Array<Long>
fun Multik.ndarrayOf(vararg items: Float): D1Array<Float>
fun Multik.ndarrayOf(vararg items: Double): D1Array<Double>
inline fun <reified T : Complex> Multik.ndarrayOf(vararg items: T): D1Array<T>
```

## Parameters

| Parameter | Type       | Description            |
|-----------|------------|------------------------|
| `items`   | `vararg T` | Elements of the array. |

**Returns:** `D1Array<T>`

## Example

<!---FUN ndarrayOf_example-->

```kotlin
val a = mk.ndarrayOf(1, 2, 3)          // D1Array<Int>: [1, 2, 3]
val b = mk.ndarrayOf(1.0, 2.0, 3.0)   // D1Array<Double>: [1.0, 2.0, 3.0]
```

<!---END-->

> For arrays with more than one dimension, use [ndarray](ndarray.md) or [d2array](d2array.md) and friends.
> {style="note"}

<seealso style="cards">
<category ref="api-docs">
  <a href="ndarray.md" summary="Create arrays from lists, collections, and primitive arrays."/>
  <a href="d1array.md" summary="Create 1D arrays with an init lambda."/>
</category>
</seealso>