# Installation

<web-summary>
The Installation guide provides comprehensive instructions on integrating the Multik library into your project.
It covers steps for adding Multik to both regular and multi-platform projects using gradle,
and offers guidance on utilizing Multik in Kotlin-Notebook, Jupyter, and Datalore environments.
</web-summary>

<card-summary>
The Installation section offers clear instructions for integrating the Multik library into your projects,
whether they are regular, multi-platform,
or executed in notebook environments like Kotlin-Notebook, Jupyter, or Datalore.
</card-summary>

<link-summary>
Learn how to add the Multik library to your project, including regular, multi-platform,
and notebook environments like Kotlin-Notebook, Jupyter, or Datalore.
</link-summary>

## Gradle

To utilize Multik in your project, the following steps need to be performed:

* Add the Maven Central Repository in your project.
* Decide the specific dependency that you require.
  You can find more information in [Engines of Multik](engines-of-multik.md).
  If you're unsure, you can use the `default` dependency.
  > The default dependency also includes the native library.
  >
  {style="warning"}
* Add the necessary dependency to your project. The available dependencies are:
    * org.jetbrains.kotlinx:multik-core:%mk_latest_version%
    * org.jetbrains.kotlinx:multik-default:%mk_latest_version%
    * org.jetbrains.kotlinx:multik-kotlin:%mk_latest_version%
    * org.jetbrains.kotlinx:multik-openblas:%mk_latest_version%

<tabs group="languages" id="main-class-set-engine-main">
<tab title="Gradle (Kotlin)" group-key="kotlin">

```kotlin
repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:multik-default:%mk_latest_version%")
}
```

</tab>
<tab title="Gradle (Groovy)" group-key="groovy">

```groovy
repositories {
    mavenCentral()
}

dependencies {
    implementation "org.jetbrains.kotlinx:multik-default:%mk_latest_version%"
}
```

</tab>
</tabs>

For multiplatform projects, add Multik in the `commonBlock`:

```kotlin
kotlin {
    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:multik-core:%mk_latest_version%")
            }
        }
    }
}
```

Alternatively, you can add Multik in the block of your required platform.
Here is an example of adding Multik for the JVM platform:

```kotlin
kotlin {
    sourceSets {
        val jvmName by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:multik-core-jvm:%mk_latest_version%")
            }
        }
    }
}
```

## Kotlin-Notebook, Jupyter, Datalore

Multik can also be used in interactive environments such
as [Kotlin Notebooks](https://kotlinlang.org/docs/data-science-overview.html#kotlin-notebook),
[Jupyter with the Kotlin kernel](https://kotlinlang.org/docs/data-science-overview.html#jupyter-kotlin-kernel),
and [Datalore](https://datalore.jetbrains.com).

To do so, use the following magic command:

```
%use multik
```

This command includes the `multik-default` dependency and all necessary imports for working with the library.

In a Kotlin-Notebook, you'll have access to the full suite of Kotlin features that you are accustomed to in IDEA.
This setup provides a convenient platform for quickly testing hypotheses and implementing ideas.

![Kotlin-Notebook with Multik](kotlin_notebook_installation.png)

To delve deeper into Kotlin Notebooks, we recommend you read the dedicated post on the JetBrains blog:
[Introducing Kotlin Notebook](https://blog.jetbrains.com/kotlin/2023/07/introducing-kotlin-notebook).

<seealso style="cards">
<category ref="ext">
  <a href="https://kotlinlang.org/docs/data-science-overview.html" 
      summary="A comprehensive list of Kotlin tools and libraries designed for data science applications.">
        Kotlin For Data Science
  </a>
  <a href="https://plugins.jetbrains.com/plugin/16340-kotlin-notebook" 
      summary="A plugin that introduces interactive Kotlin notebooks">
        Kotlin Notebook plugin
  </a>
  <a href="https://github.com/Kotlin/kotlin-jupyter" 
      summary="GitHub repository for the Kotlin kernel that can be integrated into Jupyter notebooks.">
        Kotlin Kernel
  </a>
  <a href="https://datalore.jetbrains.com" 
      summary="An interactive web-based editor designed for notebooks by JetBrains.">
        Datalore
  </a>
</category>
</seealso>