# Overview

<!---IMPORT samples.docs.Overview-->

<web-summary>
Discover Multik, a powerful multiplatform library for multidimensional array operations in Kotlin.
Explore its key principles, unique features such as static typing and dimensional consistency,
learn about its robust architecture, and delve into its performance considerations.
Understand how Multik enhances mathematical computations in Kotlin with its specialized capabilities.
</web-summary>

<card-summary>
Dive into the core of Multik.
Understand its features, architecture, performance benefits, and seamless integration with Kotlin's standard library.
</card-summary>

<link-summary>
Explore the essence of Multik: its capabilities, principles, architectural aspects,
and performance for multidimensional array computations.
</link-summary>

Multik, a multiplatform library, is a powerful tool for handling multidimensional arrays in a versatile and efficient
manner. It equips developers with high-speed mathematical and arithmetic operations, comprehensive linear algebra
functionalities, and robust statistical procedures, along with an arsenal of transformation and sorting utilities.

Multik has been developed adhering to the following principles:

* An intuitive and straightforward API
* A statically typed API for error prevention and clarity
* Optimized for high performance
* Flexibility and effortless startup for ease of use

In the Multik library, the pivotal element is the multidimensional array. It encapsulates homogeneous data in a
versatile multidimensional framework, enabling streamlined computations and transformations. In Kotlin, developers can
create their own multidimensional arrays and perform a spectrum of data manipulations using iterative procedures. Here's
a simple illustration of managing a matrix using Kotlin's standard library and Multik:

<compare first-title="stdlib Kotlin" second-title="Multik" type="top-bottom">

<!---FUN create_kotlin_matrix-->

```kotlin
val a = listOf(listOf(1, 2, 3), listOf(4, 5, 6))
val b = listOf(listOf(7, 8, 9), listOf(10, 11, 12))
val c = MutableList(2) { MutableList(3) { 0 } }
for (i in a.indices) {
    for (j in a.first().indices) {
        c[i][j] = a[i][j] * b[i][j]
    }
}
println(c) //[[7, 16, 27], [40, 55, 72]]
```

<!---END-->

<!---FUN create_multik_matrix-->

```kotlin
val a = mk.ndarray(mk[mk[1, 2, 3], mk[4, 5, 6]])
val b = mk.ndarray(mk[mk[7, 8, 9], mk[10, 11, 12]])
val c = a * b
println(c)
/*
[[7, 16, 27],
 [40, 55, 72]]
 */
```

<!---END-->

</compare>

While Kotlin does provide a robust API for handling such tasks, employing Multik minimizes code complexity and enhances
the readability of the implementation.
Moreover, Multik's performance is considerably superior compared to Kotlin's standard
library.

One unique aspect of Multik's multidimensional arrays is their support for static typing and dimensions. This feature
enables detection of a data type and dimensional inconsistencies at the compile-time itself, ensuring the production of
more robust and dependable code.

```kotlin
val a = mk.ndarray(mk[1, 2, 3, 4, 5, 6])
val b = mk.ndarray(mk[0.2, 2.2, 3.7, 4.74, 5.9, 6.17])
val c = mk.ndarray(mk[mk[5, 2, 8], mk[4, 3, 9]])

val d = a + b // compile error (type mismatch)
val e = a dot c // compile error (dimension mismatch)
```

## Architecture {id="overview-architecture"}

Multik provides a unified API across different implementations. One implementation is purely based on Kotlin, ensuring
maximum platform compatibility and easy installation. Another one incorporates C++/Fortran libraries for lightning-fast
computations. Moreover, Multik caters to scenarios where users are unsure of the optimal implementation for their tasks.
In such cases, a default implementation dynamically selects the best fit based on the nature of the task at hand. These
implementations are organized into modules as depicted in the following diagram:

![Architecture](overview-architecture.png) {width="700"}

The multiplatform technology in Kotlin is being rapidly advanced, greatly simplifying cross-platform project
development. With the native multiplatform support in Multik, you can effortlessly share your mathematical computations
across various platforms.

> * More details about the multiplatform technology can be found in
    > the [official Kotlin documentation](https://kotlinlang.org/docs/multiplatform.html).
> * You can check out the platforms supported by Multik in [this link](supported-platforms.md).

## Performance {id="overview-performance"}

Performance is a pivotal factor in Multik, but what gives it this edge?
At the core of a ndarray lies a primitive array.
When you create a three-dimensional array or a matrix, under the hood, there is a single primitive array of the
corresponding type. This means that your data resides as a contiguous memory block, resulting in faster iteration-based
operations compared to an array of objects scattered across different memory locations.

But that's not all. While the JVM is continually improving and getting faster, many users still use older versions.
That's why we've crafted a special implementation where operations are pushed to native code and the time-tested
OpenBLAS library is used. This implementation truly offers stunning performance.

