package org.jetbrains.multik.jni

import java.nio.Buffer

/**
 * JNI object stores native function.
 */
object Basic {
    external fun allocate(data: Buffer): Long
    external fun delete(handle: Long): Unit
}