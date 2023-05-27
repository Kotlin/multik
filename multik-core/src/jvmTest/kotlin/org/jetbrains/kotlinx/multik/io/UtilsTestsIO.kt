package org.jetbrains.kotlinx.multik.io

class UtilsTestsIO

internal fun testResource(resourcePath: String): String =
    UtilsTestsIO::class.java.classLoader.getResource(resourcePath)!!.path

internal fun testCsv(csvName: String) = testResource("data/csv/$csvName.csv")

internal fun testNpy(npyName: String) = testResource("data/npy/$npyName.npy")

internal fun testNpz(npzName: String) = testResource("data/npy/$npzName.npz")