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

    private val _engines: MutableMap<String, EngineType> = mutableMapOf(
        "JVM" to JvmEngineType,
        "NATIVE" to NativeEngineType
    )

    public val engines: Map<String, EngineType>
        get() = _engines

    public fun addEngine(type: EngineType) {
        _engines.putIfAbsent(type.name, type)
    }

    public fun setEngine(type: EngineType) {
        if (type.name in engines)
            Engine.setDefaultEngine(type)
    }

    public val math: Math get() = Engine.getMath()
    public val linalg: LinAlg get() = Engine.getLinAlg()
    public val stat: Statistics get() = Engine.getStatistics()

    public operator fun <T> get(vararg elements: T): List<T> = elements.toList()
}
