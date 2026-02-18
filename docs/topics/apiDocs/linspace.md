# linspace

<!---IMPORT samples.docs.apiDocs.ArrayCreation-->

<web-summary>
API reference for mk.linspace() — create a 1D array of evenly spaced values over a closed interval.
</web-summary>

<card-summary>
Create evenly spaced values in the closed interval [start, stop].
</card-summary>

<link-summary>
mk.linspace() — evenly spaced values in [start, stop].
</link-summary>

## Signatures

```kotlin
inline fun <reified T : Number> Multik.linspace(
    start: Int, stop: Int, num: Int = 50
): D1Array<T>

inline fun <reified T : Number> Multik.linspace(
    start: Double, stop: Double, num: Int = 50
): D1Array<T>
```

## Parameters

| Parameter | Type              | Default | Description                                |
|-----------|-------------------|---------|--------------------------------------------|
| `start`   | `Int` or `Double` | —       | Start of the interval (inclusive).         |
| `stop`    | `Int` or `Double` | —       | End of the interval (inclusive).           |
| `num`     | `Int`             | `50`    | Number of values to generate. Must be > 0. |

**Returns:** `D1Array<T>` with `num` evenly spaced values from `start` to `stop` inclusive.

## Example

<!---FUN linspace_example-->

```kotlin
val a = mk.linspace<Double>(0, 1, 5)
// [0.0, 0.25, 0.5, 0.75, 1.0]

val b = mk.linspace<Float>(0.0, 10.0, 3)
// [0.0, 5.0, 10.0]
```

<!---END-->

## linspace vs arange

|               | `linspace`                              | `arange`                            |
|---------------|-----------------------------------------|-------------------------------------|
| **Endpoint**  | Included (`stop` is the last element)   | Excluded (`stop` is not included)   |
| **Control**   | Specify number of points                | Specify step size                   |
| **Precision** | Last element guaranteed to equal `stop` | May accumulate floating-point drift |

> Prefer `linspace` when you need a specific number of evenly distributed points. Use [arange](arange.md) when you need
> a specific step size.
> {style="tip"}

<seealso style="cards">
<category ref="api-docs">
  <a href="arange.md" summary="Create arrays with a fixed step in [start, stop)."/>
  <a href="d1array.md" summary="Create 1D arrays with an init lambda."/>
</category>
</seealso>