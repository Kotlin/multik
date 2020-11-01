# MultiK

Multidimensional array library for Kotlin.

## Modules
* multik-api &mdash; contains ndarrays, methods called on them and [math] and [linalg] interfaces.
* multik-jvm &mdash; implementation of [math] and [linalg] interfaces on JVM.
* multik-native &mdash; implementation of [math] and [linalg] interfaces in native code using OpenBLAS.

## Using in your projects


## Examples

#### Create

```kotlin
val array = intArrayOf(1, 2, 3, 4, 5, 6)
mk.ndarray(array) // Creates ndarray [1, 2, 3, 4, 5, 6] of dimension 1 and shape (6)

// Creates ndarray
// [[1, 2, 3],
// [4, 5, 6]] of dimension 2 and shape (2, 3)
mk.ndarray(array, 2, 3)

// [[1.0, 2.0, 3.0],
// [4.0, 5.0, 6.0]]
mk.ndarray(mk[mk[1f, 2f, 3f], mk[3f, 2f, 1f]])

// [[[0, 1, 4],
// [9, 16, 25]],
// 
// [[36, 49, 64],
// [81, 100, 121]]]
mk.d3array(2, 2, 3) { it * it }

mk.arange<Long>(10, 25, 5) // [10, 15, 20]

mk.linspace<Double>(0, 2, 9) // [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

// [[1, 2],
// [3, 4]
mk.ndarray<Int, D2>(setOf(1, 2, 3, 4), intArrayOf(2, 2)) //
```

#### Indexing/Iterating

```kotlin
val array = floatArrayOf(1.5f,2.3f,3.1f,4.7f,5.9f,6.1f,7f,8f,9f,10f,11f,12f)
val a = mk.ndarray(array, 2, 2, 3)

a[1, 1, 0] // 10.0

a[1, 0] // [7.0, 8.0, 9.0]

// for n-dimensional
val b = a.asDNArray()

// [[1.5, 2.3, 3.1],
// [4.7, 5.9, 6.1]]
b.V[0]

// [[7.0, 8.0, 9.0],
// [10.0, 11.0, 12.0]]
b.V[1]

for (el in a) {
    print("$el, ") // 1.5, 2.3, 3.1, 4.7, 5.9, 6.1, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 
}

// for n-dimensional
println(b.multiIndices) // (0, 0, 0)..(2, 2, 3)
for (index in b.multiIndices) {
    print("${b[index]}, ") // 1.5, 2.3, 3.1, 4.7, 5.9, 6.1, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
}
```

#### Arithmetic operations

```kotlin
val a = mk.ndarray(mk[1.0, 2.0, 3.0])
val b = mk.ndarray(mk[4.0, 5.0, 6.0])

a + b // [5.0, 7.0, 9.0]

a += 1.0 // a = [2.0, 3.0, 4.0]

a - b // [-2.0, -2.0, -2.0]

a / b // [0.5, 0.6, 0.6666666666666666]

a * b // [8.0, 15.0, 24.0]
```

#### Mathematics 

```kotlin
val a = mk.ndarray(mk[mk[mk[1.5f,2.3f,3.1f], mk[4.7f,5.9f,6.1f]], mk[mk[7f,8f,9f], mk[10f,11f,12f]]])

mk.math.argMax(a) // 11

mk.math.exp(a)
// [[[4.4816890703380645, 9.97418197920865, 22.197949164480132],
// [109.94715148136659, 365.0375026780162, 445.85772756220854]],
//
// [[1096.6331584284585, 2980.9579870417283, 8103.083927575384],
// [22026.465794806718, 59874.14171519782, 162754.79141900392]]]

mk.math.cumSum(a) // [1.5, 3.8, 6.8999996, 11.599999, 17.5, 23.6, 30.6, 38.6, 47.6, 57.6, 68.6, 80.6]

mk.math.max(a) // 12.0
mk.math.min(a) // 1.5
mk.math.sum(a) // 80.6

val matrix = mk.ndarray(mk[mk[1f, 2f, 3f], mk[4f, 5f, 6f], mk[7f, 8f, 9f]])
mk.linalg.dot(matrix, matrix)
// [[30.0, 36.0, 42.0],
// [66.0, 81.0, 96.0],
// [102.0, 126.0, 150.0]]
```

#### Inplace

```kotlin
val a = mk.linspace<Float>(0, 1, 10)
val b = mk.linspace<Float>(8, 9, 10)

a.inplace { 
    math { 
        (this - b) * b
         abs()
    }
}
// a = [64.0, 64.88888, 65.77778, 66.66666, 67.55556, 68.44444, 69.333336, 70.22222, 71.111115, 72.0]
```

#### Iterable

```kotlin
val a = mk.ndarray(mk[mk[mk[1.5f,2.3f,3.1f], mk[4.7f,5.9f,6.1f]], mk[mk[7f,8f,9f], mk[10f,11f,12f]]])

a.filter { it < 7 } // [1.5, 2.3, 3.1, 4.7, 5.9, 6.1]

a.map { (it * it).toInt() }
// [[[2, 5, 9],
// [22, 34, 37]],
//
// [[49, 64, 81],
// [100, 121, 144]]]

mk.arange<Long>(50).dropWhile { it < 45 } // [45, 46, 47, 48, 49]

mk.d3array(2, 2, 2) { it }.groupNdarrayBy { it % 2 } // {0=[0, 2, 4, 6], 1=[1, 3, 5, 7]}
```

## Building
Multik uses blas for implementing algebraic operations. Therefore, you would need a C ++ compiler.
To build, run `./gradlew assemble`

## Testing
`./gradlew test`

## Benchmarks

The benchmark code be found in `src/jmh` folder. To run the benchmarks, run following commands:
```
./gradlew assemble benchmarkJar
java -jar ./build/libs/multik-benchmark.jar
```

## Contributing
There is an opportunity to contribute to the project:
1. Implement [math](multik-api/src/main/kotlin/org/jetbrains/multik/api/Math.kt) and [linalg](multik-api/src/main/kotlin/org/jetbrains/multik/api/LinAlg.kt) interfaces.
2. Create your own engine successor from [Engine](multik-api/src/main/kotlin/org/jetbrains/multik/api/Engine.kt), for example - [JvmEngine](multik-jvm/src/main/kotlin/org/jetbrains/multik/jvm/JvmEngine.kt).
3. Use [mk.addEngine](https://github.com/devcrocod/multik/blob/972b18cfd2952abd811fabf34461d238e55c5587/multik-api/src/main/kotlin/org/jetbrains/multik/api/Multik.kt#L23) and [mk.setEngine](https://github.com/devcrocod/multik/blob/972b18cfd2952abd811fabf34461d238e55c5587/multik-api/src/main/kotlin/org/jetbrains/multik/api/Multik.kt#L27)
to use your implementation.