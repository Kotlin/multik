# The complex double dot builds its result in multik_jni by calling this factory by name (ComplexDouble.cpp).
# R8 can't see that call: without the rule it inlines the factory and drops the class, and the JNI lookup aborts the app.
-keep,includedescriptorclasses class org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble_jvmKt {
    public static org.jetbrains.kotlinx.multik.ndarray.complex.ComplexDouble ComplexDouble(double, double);
}

# JNI binds native methods by class and method name. proguard-android-optimize.txt keeps them as well,
# but the AAR must not rely on the app using it.
-keepclasseswithmembernames,includedescriptorclasses class org.jetbrains.kotlinx.multik.openblas.** {
    native <methods>;
}
