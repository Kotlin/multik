# Copilot Code Review Instructions

You are reviewing pull requests for **Multik** — a Kotlin Multiplatform ndarray library (`org.jetbrains.kotlinx`).

## Public API changes

- **No data classes in public API.** Adding a property later changes the constructor, `copy()`, and `componentN()` signatures,
    breaking binary compatibility. Suggest a regular class with an explicit `toString`.
- **No new arguments on existing public functions**, even with default values — this breaks binary compatibility on JVM.
    Suggest adding a manual overload instead.
- **No widening or narrowing of return types** on existing public functions — both break binary compatibility.
- **No boolean parameters** in public API. Suggest separate named functions or an enum type.
- **No mutable collections or arrays in return types.** Return read-only collections; if an array must be returned,
    ensure a defensive copy is made.
- New unstable APIs must be annotated with `@RequiresOptIn` (e.g. `@ExperimentalMultikApi`).
- Deprecated APIs must use `@Deprecated` with `message`, `replaceWith`, and appropriate `level`.
- Validate inputs with `require()` and object state with `check()`. Error messages should include the offending value.

## Naming and consistency

- Parameter names must be consistent with existing API.
- Nullable-returning variants use the `OrNull` suffix (e.g. `firstOrNull`).
  Exception-catching variants use the `Catching` suffix.
- Test functions are named `testXxx` — no backticks. Test classes end with `*Test` or `*Tests`.

## Code style

- Extension functions for non-core functionality. Core interfaces (`MultiArray`, `MutableMultiArray`) should stay minimal;
  extras go into `ndarray.operations` as extensions.
- Sealed types for closed hierarchies (dimension types, etc.) — flag open classes/interfaces where sealed is more appropriate.

## Multiplatform

- APIs should be placed in the broadest possible source set (`commonMain` preferred).
- `expect`/`actual` implementations must behave identically across platforms.
- Flag platform-specific code in `commonMain` or missing `actual` declarations.

## Engine operations

When a new operation is added to `Math`, `LinAlg`, or `Statistics`:
- The signature must be in the interface in `multik-core`.
- Implementation must exist in `multik-kotlin` (all platforms). Implementation in `multik-openblas` (JVM + desktop Native) is recommended.
- Tests must cover both engines.

## Documentation

- New public API must have KDoc. The description should explain *what* the function does, not restate the signature.
- Lambda parameters must document: exception behavior, threading guarantees, sequencing.
- Valid input ranges, invalid input behavior, and thrown exceptions must be specified.
- If doc samples in `multik-core/src/commonTest/kotlin/samples/docs/` are affected, they must stay in sync with `docs/topics/`.

## Performance red flags

- **Unnecessary copies.**
- **Boxing in tight loops.** `NDArray` is backed by primitive arrays. Flag usage of generic `Array<T>`, `List<T>`, or `Iterable<T>` where a typed `MemoryView` or primitive array should be used instead.
- **Redundant allocations.** Flag `IntArray`/`DoubleArray`/etc. created per-call when they could be reused or preallocated (e.g. strides, shape arrays in repeated operations).
- **O(n) where O(1) is possible.** Accessing `NDArray.size`, `shape`, `strides` is O(1). Flag code that recomputes these values (e.g. iterating to count elements).
- **JNI boundary overhead.** Each JNI call has overhead. Flag code in `multik-openblas` that crosses the JNI boundary per-element instead of passing entire arrays to native functions.
- **Stride-unaware iteration.** When iterating non-contiguous views, the iterator must walk strides. Flag code that assumes contiguous memory layout without checking `consistent`.
- **Inefficient loops.** Flag nested loops that can be replaced with bulk array operations, unnecessary repeated scans, or O(n^2) patterns where O(n) is achievable.
- **Memory leaks and resource cleanup.** Ensure `Closeable` resources use `use {}`. Flag views holding large backing arrays when only a small slice is needed — suggest `deepCopy()` to release the original.
