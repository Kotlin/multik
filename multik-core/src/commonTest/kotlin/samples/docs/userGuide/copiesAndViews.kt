package samples.docs.userGuide

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.data.slice
import kotlin.test.Test

class CopiesAndViews {
    @Test
    fun creating_views() {
        // SampleStart
        val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        val v = a[0] // view of the first row

        println(v) // [1, 2]

        a[0, 0] = 99
        println(v) // [99, 2]
        // SampleEnd
    }

    @Test
    fun view_with_slice() {
        val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        // SampleStart
        val s = a.slice<Int, D2, D2>(0..1, axis = 1)
        println(s.base != null) // true for views
        // SampleEnd
    }

    @Test
    fun copy_and_deepCopy() {
        val a = mk.ndarray(mk[mk[1, 2], mk[3, 4]])
        // SampleStart
        val view = a[0]
        val c1 = view.copy()
        val c2 = view.deepCopy()

        a[0, 0] = 111
        println(view) // changed: [111, 2]
        println(c1)   // unchanged: [1, 2]
        println(c2)   // unchanged: [1, 2]
        // SampleEnd
    }
}