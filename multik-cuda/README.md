# Multik CUDA module
This module provides implementation of Multik API accelerated with CUDA computations.

## Dependencies
* `CUDA Toolkit` - [Installation guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)
* `JCuda`, `JCublas` - Java bindings for CUDA and cuBLAS. [Website](http://www.jcuda.org/)
* `Kotlin Logging` - logging framework for Kotlin. [Repository](https://github.com/MicroUtils/kotlin-logging)

## Architecture
Current implementation restricts user to use only immutable NDArrays.

Current architecture encourages doing CPU reads as few times as possible.
CPU reads are: getting individual elements of the array, slicing the array (for now), deepCoping/cloning the array.  

There are 2 major memory optimization:
* GPU memory caching — on first use of an array in GPU it is copied to GPU and saved to GpuCache.
Next usages of this array in GPU operations will get the VRAM address of this array from cache (if it hasn't been cleaned yet).
GpuCache is an LRU cache, it uses `java.lang.ref.Cleaner` inside and depends on work of Garbage Collector.
* Lazy copy to RAM — arrays are copied to RAM lazily only on demand (CPU read). Because of that try minimizing the amount of CPU reads.  

CUDA module supports JVM multithreading — executing multiple operations is safe as far as there is now JVM read/write conflicts. 

To use CUDA implementation context must be initialized and then deinitialized.
CUDA context is thread local, so it must be initialized and then deinitialized on every thread where operations will take place.

You can initialize/deinitialize context like so:
```kotlin
CudaEngine.initCuda()       // initialize context
...
CudaEngine.deinitCuda()     // deinitialize context
```
or like so:
```kotlin
CudaEngine.runWithCuda {    // initialize context
    ...
}                           // deinitialize context
```

## Example
```kotlin
CudaEngine.runWithCuda {
    val mat1 = mk.ndarray(
        mk[
                mk[1.0, 1.1, 1.2],
                mk[1.3, 1.4, 1.5],
                mk[1.6, 1.7, 1.8],
        ]
    )

    var array = mat1

    repeat(5) {
        array = CudaLinAlg.dot(array, array)
    }

    println(array)
}
```
