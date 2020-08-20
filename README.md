# MultiK

Multidimensional array library for Kotlin.

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