# Security

## Supported Versions

Security updates are applied to the latest released version of Multik.

A fix will be shipped with the next patch or minor release.

## Reporting a Vulnerability

Instructions for reporting a vulnerability can be found on the [Security page](https://kotlinlang.org/docs/security.html).

## Security Considerations

Multik includes a native module (`multik-openblas`) that uses C++/JNI bindings to OpenBLAS.
If you discover a vulnerability related to the native code, memory handling, or JNI bridge,
please report it using the methods above.