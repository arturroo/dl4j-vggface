lazy val root = (project in file(".")).
  settings(
    name := "dl4j-vggface-gh",
    version := "0.9",
    scalaVersion := "2.11.8"
  )

libraryDependencies ++= Seq(
  "args4j" % "args4j" % "2.33",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1",
  "org.deeplearning4j" % "deeplearning4j-zoo" % "0.9.1",
  "org.nd4j" % "nd4j-native-platform" % "0.9.1"
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)
assemblyJarName in assembly := "dl4j-vggface-gh.jar"



