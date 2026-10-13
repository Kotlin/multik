package org.jetbrains.kotlinx.multik.builds

import org.gradle.api.DefaultTask
import org.gradle.api.file.ConfigurableFileCollection
import org.gradle.api.file.DirectoryProperty
import org.gradle.api.file.FileSystemOperations
import org.gradle.api.provider.Property
import org.gradle.api.tasks.Input
import org.gradle.api.tasks.InputFiles
import org.gradle.api.tasks.OutputDirectory
import org.gradle.api.tasks.PathSensitive
import org.gradle.api.tasks.PathSensitivity
import org.gradle.api.tasks.TaskAction
import javax.inject.Inject

/**
 * Syncs [files] into `<outputDir>/<subdirectory>/`.
 *
 * Unlike the built-in `Sync` task, the output is a [DirectoryProperty],
 * which AGP's `addGeneratedSourceDirectory` requires. Missing files are skipped.
 */
abstract class SyncIntoSubdirectory : DefaultTask() {
    @get:InputFiles
    @get:PathSensitive(PathSensitivity.NAME_ONLY)
    abstract val files: ConfigurableFileCollection

    @get:Input
    abstract val subdirectory: Property<String>

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @get:Inject
    abstract val fs: FileSystemOperations

    @TaskAction
    fun sync() {
        fs.sync {
            from(files) { into(subdirectory.get()) }
            into(outputDir)
        }
    }
}
