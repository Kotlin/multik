[![Kotlin Alpha](https://kotl.in/badges/alpha.svg)](https://kotlinlang.org/docs/components-stability.html)
[![JetBrains incubator project](https://jb.gg/badges/incubator.svg)](https://confluence.jetbrains.com/display/ALL/JetBrains+on+GitHub)
[![Maven Central](https://img.shields.io/maven-central/v/org.jetbrains.kotlinx/multik-core)](https://central.sonatype.com/artifact/org.jetbrains.kotlinx/multik-core)
[![GitHub license](https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg?style=flat)](https://www.apache.org/licenses/LICENSE-2.0)

# Multik

Multidimensional array library for Kotlin.

Multik provides N-dimensional arrays with type-safe dimensions, math operations, linear algebra, and statistics.
It works across JVM, JS, WasmJS, iOS, and desktop native targets via Kotlin Multiplatform,
with optional OpenBLAS acceleration for high performance.

## Modules

| Module              | Description                                                                                           |
|---------------------|-------------------------------------------------------------------------------------------------------|
| **multik-core**     | Core ndarray types, the `mk` entry point, and Math/LinAlg/Statistics API interfaces. All platforms.   |
| **multik-default**  | Combines `multik-kotlin` and `multik-openblas` for optimal performance on every platform.             |
| **multik-kotlin**   | Pure Kotlin implementation. JVM, JS, WasmJS, iOS, and desktop native.                                 |
| **multik-openblas** | Native implementation backed by OpenBLAS via C++/JNI. JVM and desktop native (macOS, Linux, Windows). |

## Installation

Latest
version: [![Maven Central](https://img.shields.io/maven-central/v/org.jetbrains.kotlinx/multik-core)](https://central.sonatype.com/artifact/org.jetbrains.kotlinx/multik-core)

### Gradle Kotlin DSL

`multik-core` provides ndarray types, creation functions, and basic operations.

For linear algebra, statistics, and math engines, add an engine dependency —
`multik-default`, `multik-kotlin`, or `multik-openblas`.
Engine dependencies transitively include `multik-core`.

`build.gradle.kts`:

```kotlin
repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:multik-default:$multikVersion")
}
```

<details>
<summary>Gradle Groovy DSL</summary>

`build.gradle`:

```groovy
repositories {
    mavenCentral()
}

dependencies {
    implementation "org.jetbrains.kotlinx:multik-default:$multikVersion"
}
```

</details>

### Kotlin Multiplatform

```kotlin
kotlin {
    sourceSets {
        commonMain {
            dependencies {
                implementation("org.jetbrains.kotlinx:multik-default:$multikVersion")
            }
        }
    }
}
```

### Jupyter Notebook

Install [Kotlin kernel](https://github.com/Kotlin/kotlin-jupyter) for
[Jupyter](https://jupyter.org/)
or just visit to [Datalore](https://datalore.jetbrains.com/).

Import stable `multik` version into notebook:

```
%use multik
```

## Supported Platforms

|       Platform        | `multik-core` | `multik-kotlin` | `multik-openblas` | `multik-default` |
|:---------------------:|:-------------:|:---------------:|:-----------------:|:----------------:|
|        **JVM**        |       ✅       |        ✅        |         ✅         |        ✅         |
|        **JS**         |       ✅       |        ✅        |         —         |        ✅         |
|      **WasmJS**       |       ✅       |        ✅        |         —         |        ✅         |
|     **linuxX64**      |       ✅       |        ✅        |         ✅         |        ✅         |
|     **mingwX64**      |       ✅       |        ✅        |         ✅         |        ✅         |
|     **macosX64**      |       ✅       |        ✅        |         ✅         |        ✅         |
|    **macosArm64**     |       ✅       |        ✅        |         ✅         |        ✅         |
|     **iosArm64**      |       ✅       |        ✅        |         —         |        ✅         |
|      **iosX64**       |       ✅       |        ✅        |         —         |        ✅         |
| **iosSimulatorArm64** |       ✅       |        ✅        |         —         |        ✅         |

> [!IMPORTANT]
> - On Ubuntu 18.04 and older, `multik-openblas` doesn't work due to older versions of **glibc**.
> - `multik-openblas` for desktop native targets (_linuxX64_, _mingwX64_, _macosX64_, _macosArm64_) is experimental and
>     unstable.
> - JVM target `multik-openblas` for Android only supports **arm64-v8a** processors.

## Quickstart

Visit [Multik documentation](https://kotlin.github.io/multik) for a detailed feature overview.

### Creating arrays

```kotlin
val a = mk.ndarray(mk[1, 2, 3])
/* [1, 2, 3] */

val b = mk.ndarray(mk[mk[1.5, 2.1, 3.0], mk[4.0, 5.0, 6.0]])
/*
[[1.5, 2.1, 3.0],
[4.0, 5.0, 6.0]]
*/

val c = mk.ndarray(mk[mk[mk[1.5f, 2f, 3f], mk[4f, 5f, 6f]], mk[mk[3f, 2f, 1f], mk[4f, 5f, 6f]]])
/*
[[[1.5, 2.0, 3.0],
[4.0, 5.0, 6.0]],

[[3.0, 2.0, 1.0],
[4.0, 5.0, 6.0]]]
*/


mk.zeros<Double>(3, 4) // create an array of zeros
/*
[[0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0],
[0.0, 0.0, 0.0, 0.0]]
*/
mk.ndarray<Float, D2>(setOf(30f, 2f, 13f, 12f), intArrayOf(2, 2)) // create an array from a collection
/*
[[30.0, 2.0],
[13.0, 12.0]]
*/
val d = mk.ndarray(
    doubleArrayOf(1.0, 1.3, 3.0, 4.0, 9.5, 5.0),
    2,
    3
) // create an array of shape(2, 3) from a primitive array
/*
[[1.0, 1.3, 3.0],
[4.0, 9.5, 5.0]]
*/
mk.d3array(2, 2, 3) { it * it } // create an array of 3 dimension
/*
[[[0, 1, 4],
[9, 16, 25]],

[[36, 49, 64],
[81, 100, 121]]]
*/

mk.d2arrayIndices(3, 3) { i, j -> ComplexFloat(i, j) }
/*
[[0.0+(0.0)i, 0.0+(1.0)i, 0.0+(2.0)i],
[1.0+(0.0)i, 1.0+(1.0)i, 1.0+(2.0)i],
[2.0+(0.0)i, 2.0+(1.0)i, 2.0+(2.0)i]]
 */

mk.arange<Long>(10, 25, 5) // create an array with elements in the interval [10, 25) with step 5
/* [10, 15, 20] */

mk.linspace<Double>(0, 2, 9) // create an array of 9 elements in the interval [0, 2]
/* [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0] */

val e = mk.identity<Double>(3) // create an identity array of shape (3, 3)
/*
[[1.0, 0.0, 0.0],
[0.0, 1.0, 0.0],
[0.0, 0.0, 1.0]]
*/

val diag = mk.diagonal(mk[2, 4, 8]) // create a diagonal array
/*
[[2, 0, 0],
[0, 4, 0],
[0, 0, 8]]
 */
```

### Array properties

```kotlin
a.shape // Array dimensions
a.size // Size of array
a.dim // object Dimension
a.dim.d // number of array dimensions
a.dtype // Data type of array elements
```

### Arithmetic operations

```kotlin
val f = b - d // subtraction
/*
[[0.5, 0.8, 0.0],
[0.0, -4.5, 1.0]]
*/

d + f // addition
/*
[[1.5, 2.1, 3.0],
[4.0, 5.0, 6.0]]
*/

b / d // division
/*
[[1.5, 1.6153846153846154, 1.0],
[1.0, 0.5263157894736842, 1.2]]
*/

f * d // multiplication
/*
[[0.5, 1.04, 0.0],
[0.0, -42.75, 5.0]]
*/
```

### Math, Linear Algebra, and Statistics

See documentation for other methods of
[mathematics](https://kotlin.github.io/multik/multik-core/org.jetbrains.kotlinx.multik.api.math/index.html),
[linear algebra](https://kotlin.github.io/multik/multik-core/org.jetbrains.kotlinx.multik.api.linalg/index.html),
[statistics](https://kotlin.github.io/multik/multik-core/org.jetbrains.kotlinx.multik.api.stat/index.html).

```kotlin
a.sin() // element-wise sin, equivalent to mk.math.sin(a)
a.cos() // element-wise cos, equivalent to mk.math.cos(a)
b.log() // element-wise natural logarithm, equivalent to mk.math.log(b)
b.exp() // element-wise exp, equivalent to mk.math.exp(b)
d dot e // dot product, equivalent to mk.linalg.dot(d, e)

mk.math.sum(c) // array-wise sum
mk.math.min(c) // array-wise minimum elements
mk.math.maxD3(c, axis = 0) // maximum value of an array along axis 0
mk.math.cumSum(b, axis = 1) // cumulative sum of the elements
mk.stat.mean(a) // mean
mk.stat.median(b) // median
```

### Copying arrays

```kotlin
val f = a.copy() // create a copy of the array and its data
val h = b.deepCopy() // create a copy of the array and copy the meaningful data
```

### Collection Operations

```kotlin
c.filter { it < 3 } // select all elements less than 3
b.map { (it * it).toInt() } // return squares
c.groupNDArrayBy { it % 2 } // group elements by condition
c.sorted() // sort elements
```

### Indexing/Slicing/Iterating

```kotlin
a[2] // select the element at the 2 index
b[1, 2] // select the element at row 1 column 2
b[1] // select row 1
b[0..1, 1] // select elements at rows 0 to 1 in column 1
b[0, 0..2..1] // select elements at row 0 in columns 0 to 2 with step 1

for (el in b) {
    print("$el, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0,
}

// for n-dimensional
val q = b.asDNArray()
for (index in q.multiIndices) {
    print("${q[index]}, ") // 1.5, 2.1, 3.0, 4.0, 5.0, 6.0,
}
```

### Inplace

```kotlin
val a = mk.linspace<Float>(0, 1, 10)
/*
a = [0.0, 0.1111111111111111, 0.2222222222222222, 0.3333333333333333, 0.4444444444444444, 0.5555555555555556,
0.6666666666666666, 0.7777777777777777, 0.8888888888888888, 1.0]
*/
val b = mk.linspace<Float>(8, 9, 10)
/*
b = [8.0, 8.11111111111111, 8.222222222222221, 8.333333333333334, 8.444444444444445, 8.555555555555555,
8.666666666666666, 8.777777777777779, 8.88888888888889, 9.0]
*/

a.inplace {
    math {
        (this - b) * b
        abs()
    }
}
// a = [64.0, 64.88888, 65.77778, 66.66666, 67.55556, 68.44444, 69.333336, 70.22222, 71.111115, 72.0]
```

## Building

### Full build (with OpenBLAS)

Requires:

* JDK 8 or higher
* `JAVA_HOME` environment variable set
* `gcc`, `g++`, `gfortran` version 8 or higher (must be the same version)

```bash
./gradlew assemble
```

### Without OpenBLAS

```bash
./gradlew assemble -x build_cmake
```

### Individual modules

```bash
./gradlew :multik-core:build
```

### Running tests

```bash
./gradlew :multik-core:jvmTest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues, pull requests, and building the project.

## License

Multik is licensed under the [Apache License 2.0](LICENSE).
