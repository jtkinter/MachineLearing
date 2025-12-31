workspace "SVM"
	architecture "x64"
	startproject "SVM"
	configurations {"Release"}

project "SVM"
	location "SVM"
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
		"%{prj.name}/vendor/Matplot++ 1.2.0/include",
		"%{prj.name}/vendor/matio/include"
	}

	libdirs
	{
		"%{prj.name}/vendor/Matplot++ 1.2.0/lib",
		"%{prj.name}/vendor/Matplot++ 1.2.0/lib/Matplot++",
		"%{prj.name}/vendor/matio/lib"
	}

	links
	{
		"nodesoup.lib",
		"matplot.lib",
		"hdf5.lib",
		"hdf5_hl.lib",
		"zlib.lib",
		"libmatio.lib"
	}

	filter "system:windows"
		cppdialect "C++17"
		cdialect "C17"
		systemversion "latest"

	filter "configurations:Release"
		optimize "Off" ---Off ---
		symbols "On" --- On ----
		functionlevellinking "On"
		runtime "Release"
		linkoptions {"/INCREMENTAL:NO"}

		buildoptions{"/Oi"} --- "/Oi" ---