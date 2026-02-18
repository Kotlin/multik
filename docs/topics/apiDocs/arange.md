# arange

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.arange() — create a 1D array of evenly spaced values
within a half-open interval [start, stop).
</web-summary>

<card-summary>
Create evenly spaced values in the half-open interval [start, stop).
</card-summary>

<link-summary>
mk.arange() — evenly spaced values in [start, stop).
</link-summary>

## Signatures

### With start

```kotlin
inline fun <reified T : Number> Multik.arange(
    start: Int, stop: Int, step: Int = 1
): D1Array<T>

inline fun <reified T : Number> Multik.arange(
    start: Int, stop: Int, step: Double
): D1Array<T>
```

### Without start (starts at 0)

```kotlin
inline fun <reified T : Number> Multik.arange(
    stop: Int, step: Int = 1
): D1Array<T>

inline fun <reified T : Number> Multik.arange(
    stop: Int, step: Double
): D1Array<T>
```

## Parameters

| Parameter | Type              | Default | Description                    |
|-----------|-------------------|---------|--------------------------------|
| `start`   | `Int`             | `0`     | Start of interval (inclusive). |
| `stop`    | `Int`             | —       | End of interval (exclusive).   |
| `step`    | `Int` or `Double` | `1`     | Spacing between values.        |

**Returns:** `D1Array<T>` with values `[start, start + step, start + 2*step, …)` up to but not including `stop`.

## Example

<!---FUN arange_example-->

```kotlin
val a = mk.arange<Int>(5)              // [0, 1, 2, 3, 4]
val b = mk.arange<Int>(2, 10, 3)       // [2, 5, 8]
val c = mk.arange<Double>(0, 1, 0.3)   // [0.0, 0.3, 0.6, 0.9]
```

<!---END-->

## Constraints

* If `start < stop`, `step` must be **positive**.
* If `start > stop`, `step` must be **negative**.
* `step` must not be zero.

> For floating-point steps, accumulated rounding may cause the last value to drift. Use [](linspace.md) when you
> need an exact endpoint.
> {style="tip"}

<seealso style="cards">
<category ref="api-docs">
  <a href="linspace.md" summary="Evenly spaced values with exact endpoint in [start, stop]."/>
  <a href="d1array.md" summary="Create 1D arrays with an init lambda."/>
  <a href="zeros.md" summary="Create zero-filled arrays."/>
</category>
</seealso>
