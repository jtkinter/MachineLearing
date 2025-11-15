workspace "DecisionTree"
	architecture "x64"
	startproject "DecisionTree"
	configurations {"Release"}

project "DecisionTree"
	location "DecisionTree"
	kind "ConsoleApp"
	language "C++"

	targetdir "bin"
	objdir "bin-int"

	files
	{
		"%{prj.name}/src/**.h",
		"%{prj.name}/src/**.cpp"
	}

	includedirs
	{
		"%{prj.name}/vendor/Matplot++ 1.2.0/include"
	}

	libdirs
	{
		"%{prj.name}/vendor/Matplot++ 1.2.0/lib",
		"%{prj.name}/vendor/Matplot++ 1.2.0/lib/Matplot++"
	}

	links
	{
		"nodesoup.lib",
		"matplot.lib"
	}

	filter "system:windows"
		cppdialect "C++17"
		cdialect "C17"
		systemversion "latest"

	filter "configurations:Release"
		optimize "On" ---Off ---
		symbols "Off" --- On ----
		functionlevellinking "On"
		runtime "Release"
		linkoptions {"/INCREMENTAL:NO"}

		buildoptions{"/Oi-"} --- "/Oi" ---