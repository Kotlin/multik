package org.jetbrains.multik.api

/**
 * Alternative names.
 */
public typealias mk = Multik


/**
 * The basic object through which calls all ndarray functions.
 */
public object Multik {
    public val engine: String? get() = Engine.getDefaultEngine()

    public fun setEngine(type: EngineType) {
        Engine.setDefaultEngine(type)
    }

    public val math: Math get() = Engine.getMath()
    public val linalg: LinAlg get() = Engine.getLinAlg()

    public operator fun <T> get(vararg elements: T): List<T> = elements.toList()
}
