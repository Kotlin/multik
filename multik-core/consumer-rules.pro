# commons-csv marks CSVPrinter.endOfRecord() with SpotBugs' @SuppressFBWarnings but declares spotbugs-annotations
# optional, so the class is absent from apps and R8 fails minified builds. The annotation has CLASS retention:
# nothing reads it at runtime.
-dontwarn edu.umd.cs.findbugs.annotations.SuppressFBWarnings
