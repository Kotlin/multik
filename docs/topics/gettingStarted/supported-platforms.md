# Supported platforms

[//]: # (TODO)
<web-summary>
Multik, the efficient multidimensional array library,
is designed for use across all platforms supported by Kotlin: JVM, Native, JS, and WASM.
This page lists all the supported target platforms and presets,
ensuring Multik's wide compatibility across different operating systems and web environments.
</web-summary>

<card-summary>
Multik, the flexible multidimensional array library, can be utilized across JVM, Native, JS, and WASM platforms, boasting wide compatibility with various target presets.
</card-summary>

<link-summary>
Check out the wide range of platforms supported by Multik, including JVM,
Native, JS, and WASM, alongside an array of target presets.
</link-summary>

Multik can be utilized across the following platforms supported by Kotlin:

* JVM
* Native
* JS
* WASM

> Using Multik in a Kotlin/Native project requires
> a [new memory manager](https://kotlinlang.org/docs/native-memory-manager.html),
> which is enabled by default starting with Kotlin 1.7.20.

The following [targets](https://kotlinlang.org/docs/multiplatform-dsl-reference.html) are supported:

<table>
<tr>
    <td>
        Target platform
    </td>
    <td>
        Target preset
    </td>
</tr>
<tr>
    <td>
        Kotlin/JVM
    </td>
    <td>
        <list>
            <li>
                <code>jvm</code>
            </li>
        </list>
    </td>
</tr>

<tr>
    <td>
        iOS
    </td>
    <td>
        <list>
            <li>
                <code>iosArm64</code>
            </li>
            <li>
                <code>iosX64</code>
            </li>
            <li>
                <code>iosSimulatorArm64</code>
            </li>
        </list>
    </td>
</tr>

<tr>
    <td>
        macOS
    </td>
    <td>
        <list>
            <li>
                <code>macosX64</code>
            </li>
            <li>
                <code>macosArm64</code>
            </li>
        </list>
    </td>
</tr>

<tr>
    <td>
        Linux
    </td>
    <td>
        <list>
            <li>
                <code>linuxX64</code>
            </li>
        </list>
    </td>
</tr>

<tr>
    <td>
        Windows
    </td>
    <td>
        <list>
            <li>
                <code>mingwX64</code>
            </li>
        </list>
    </td>
</tr>

<tr>
    <td>
        JS
    </td>
    <td>
        <list>
            <li>
                <code>js</code>
            </li>
        </list>
    </td>
</tr>

<tr>
    <td>
        WASM
    </td>
    <td>
        <list>
            <li>
                <code>wasm</code>
            </li>
        </list>
    </td>
</tr>
</table>

> For a more comprehensive understanding of supported platforms, refer to
> the [Multik on Different Platforms](multik-on-different-platforms.md) section.
>
{style="note"}

<seealso>
<category ref="user-guide">
<a href="multik-on-different-platforms.md" summary="Explore Multik's compatibility and usage across various platforms">Multik on Different Platforms</a>
<a href="multik-on-JVM.md" summary="Learn about the usage and advantages of Multik on the JVM platform.">Multik on JVM</a>
<a href="multik-on-JavaScript.md" summary="Discover how Multik can enhance your JavaScript projects.">Multik on JavaScript</a>
<a href="multik-on-WASM.md" summary="Uncover the power of Multik in WebAssembly (WASM) environments.">Multik on WASM</a>
<a href="multik-on-mobile.md" summary="Find out how to leverage Multik for mobile development on Android and iOS.">Multik on Mobile (Android, iOS)</a>
<a href="multik-on-desktop.md" summary="Understand how to use Multik for desktop application development.">Multik on Desktop</a>
</category>
</seealso>