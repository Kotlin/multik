package org.jetbrains.kotlinx.multik_kotlin

import org.jetbrains.kotlinx.multik.api.mk
import java.util.concurrent.CyclicBarrier
import java.util.concurrent.atomic.AtomicReference
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertNull

class EngineInitializationConcurrencyTest {

    @Test
    fun testConcurrentEngineAccess() {
        val threadCount = 16
        val barrier = CyclicBarrier(threadCount)
        val error = AtomicReference<Throwable?>(null)
        val engines = Array<String?>(threadCount) { null }

        val threads = List(threadCount) { i ->
            Thread {
                try {
                    barrier.await()
                    engines[i] = mk.engine
                } catch (e: Throwable) {
                    error.compareAndSet(null, e)
                }
            }
        }

        threads.forEach { it.start() }
        threads.forEach { it.join() }

        assertNull(error.get(), "Concurrent engine access threw: ${error.get()}")
        val distinct = engines.filterNotNull().distinct()
        assertEquals(1, distinct.size, "All threads should resolve the same engine, got: $distinct")
    }

    @Test
    fun testConcurrentGetMath() {
        val threadCount = 16
        val barrier = CyclicBarrier(threadCount)
        val error = AtomicReference<Throwable?>(null)
        val results = Array<Any?>(threadCount) { null }

        val threads = List(threadCount) { i ->
            Thread {
                try {
                    barrier.await()
                    results[i] = mk.math
                } catch (e: Throwable) {
                    error.compareAndSet(null, e)
                }
            }
        }

        threads.forEach { it.start() }
        threads.forEach { it.join() }

        assertNull(error.get(), "Concurrent getMath() threw: ${error.get()}")
        results.forEach { assertNotNull(it, "Every thread should get a non-null Math instance") }
    }

    @Test
    fun testConcurrentGetLinAlg() {
        val threadCount = 16
        val barrier = CyclicBarrier(threadCount)
        val error = AtomicReference<Throwable?>(null)
        val results = Array<Any?>(threadCount) { null }

        val threads = List(threadCount) { i ->
            Thread {
                try {
                    barrier.await()
                    results[i] = mk.linalg
                } catch (e: Throwable) {
                    error.compareAndSet(null, e)
                }
            }
        }

        threads.forEach { it.start() }
        threads.forEach { it.join() }

        assertNull(error.get(), "Concurrent getLinAlg() threw: ${error.get()}")
        results.forEach { assertNotNull(it, "Every thread should get a non-null LinAlg instance") }
    }

    @Test
    fun testConcurrentGetStatistics() {
        val threadCount = 16
        val barrier = CyclicBarrier(threadCount)
        val error = AtomicReference<Throwable?>(null)
        val results = Array<Any?>(threadCount) { null }

        val threads = List(threadCount) { i ->
            Thread {
                try {
                    barrier.await()
                    results[i] = mk.stat
                } catch (e: Throwable) {
                    error.compareAndSet(null, e)
                }
            }
        }

        threads.forEach { it.start() }
        threads.forEach { it.join() }

        assertNull(error.get(), "Concurrent getStatistics() threw: ${error.get()}")
        results.forEach { assertNotNull(it, "Every thread should get a non-null Statistics instance") }
    }

    @Test
    fun testConcurrentMixedAccess() {
        val threadCount = 16
        val barrier = CyclicBarrier(threadCount)
        val error = AtomicReference<Throwable?>(null)

        val threads = List(threadCount) { i ->
            Thread {
                try {
                    barrier.await()
                    when (i % 4) {
                        0 -> mk.engine
                        1 -> mk.math
                        2 -> mk.linalg
                        3 -> mk.stat
                    }
                } catch (e: Throwable) {
                    error.compareAndSet(null, e)
                }
            }
        }

        threads.forEach { it.start() }
        threads.forEach { it.join() }

        assertNull(error.get(), "Concurrent mixed access threw: ${error.get()}")
    }
}
