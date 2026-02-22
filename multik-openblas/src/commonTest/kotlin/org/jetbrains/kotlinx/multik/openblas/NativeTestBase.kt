package org.jetbrains.kotlinx.multik.openblas

import kotlin.test.BeforeTest

abstract class NativeTestBase {

    protected lateinit var data: DataStructure

    @BeforeTest
    fun setUp() {
        libLoader("multik_jni").manualLoad()
        data = DataStructure(42)
        extraSetUp()
    }

    protected open fun extraSetUp() {}
}
