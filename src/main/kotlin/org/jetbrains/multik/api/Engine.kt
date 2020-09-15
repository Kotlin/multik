package org.jetbrains.multik.api

import java.util.*
import java.util.concurrent.ConcurrentHashMap

public enum class EngineType(name: String) {
    JVM("jvm"), NATIVE("native")
}


public abstract class Engine {

    internal abstract val name: String

    internal abstract val type: EngineType

    protected val engines: MutableMap<EngineType, Engine> = ConcurrentHashMap<EngineType, Engine>()

    protected var defaultEngine: EngineType? = null

    protected fun loadEngine() {
        val loaders: ServiceLoader<EngineProvider> = ServiceLoader.load(EngineProvider::class.java)
        for (engineProvider in loaders) {
            val engine = engineProvider.getEngine()
            if (engine != null) {
                engines[engine.type] = engine
            } //else {
            // need exception?
            //}
        }

//        if (engines.isEmpty()) {
        // need exception?
//        }
    }


    public abstract fun getMath(): Math
    public abstract fun getLinAlg(): LinAlg

    internal companion object : Engine() {

        init {
            loadEngine()
        }

        override val name: String
            get() = TODO("Not yet implemented")

        override val type: EngineType
            get() = TODO("Not yet implemented")

        internal fun getDefaultEngine(): String? = defaultEngine?.name

        internal fun setDefaultEngine(type: EngineType) {
            defaultEngine = type
        }

        override fun getMath(): Math {
            if (engines.isEmpty() || defaultEngine == null) throw Exception("") //TODO
            return engines[defaultEngine]?.getMath() ?: throw Exception()
        }

        override fun getLinAlg(): LinAlg {
            if (engines.isEmpty() || defaultEngine == null) throw Exception("") //TODO
            return engines[defaultEngine]?.getLinAlg() ?: throw Exception()
        }
    }
}

internal interface EngineProvider {
    fun getEngine(): Engine?
}