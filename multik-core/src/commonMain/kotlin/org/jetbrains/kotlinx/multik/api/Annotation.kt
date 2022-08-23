package org.jetbrains.kotlinx.multik.api

/**
 * Marks declarations that are still **experimental** in multik, which means that the design of the
 * corresponding declarations has open issues which may (or may not) lead to their changes in the future.
 * Roughly speaking, there is a chance that those declarations will be deprecated in the near future or
 * the semantics of their behavior may change in some way that may break some code.
 */
@MustBeDocumented
@Target(AnnotationTarget.CLASS, AnnotationTarget.PROPERTY, AnnotationTarget.FUNCTION, AnnotationTarget.TYPEALIAS)
@RequiresOptIn(level = RequiresOptIn.Level.WARNING)
public annotation class ExperimentalMultikApi