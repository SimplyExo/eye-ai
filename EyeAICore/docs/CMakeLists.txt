set(EYE_AI_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")

# this is needed to suppress this warning from doxygen caused by some kind of bug:
#    /eye-ai/EyeAICore/README.md:1: warning: No matching directory found for command \dir /eye-ai/EyeAICore/
set(CMAKE_CURRENT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../")

find_package(Doxygen)

if (DOXYGEN_FOUND)
	include(FetchContent)
	message(STATUS "Downloading doxygen-awesome-css...")
	FetchContent_Declare(
		doxygen-awesome-css
		URL "https://github.com/jothepro/doxygen-awesome-css/archive/refs/tags/v2.3.4.zip"
	)
	FetchContent_MakeAvailable(doxygen-awesome-css)
	FetchContent_GetProperties(doxygen-awesome-css SOURCE_DIR AWESOME_CSS_DIR)

	set(EYE_AI_FAVICON_PATH "${EYE_AI_CORE_DIR}/../EyeAIIcon/favicon.ico")

	# Doxyfile configuration
	set(DOXYGEN_PROJECT_NAME "EyeAI")
	set(DOXYGEN_PROJECT_ICON ${EYE_AI_FAVICON_PATH})
	set(DOXYGEN_PROJECT_LOGO "${EYE_AI_CORE_DIR}/../EyeAIIcon/eye-ai-logo-export.webp")
	set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/docs")
	set(DOXYGEN_FILE_PATTERNS *.cpp *.hpp *.md *.dox)
	set(DOXYGEN_RECURSIVE YES)
	set(DOXYGEN_EXTRACT_ALL YES)
	set(DOXYGEN_QUIET YES)
	set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)
	set(DOXYGEN_ENABLE_PREPROCESSING YES)
	set(DOXYGEN_MACRO_EXPANSION YES)
	set(DOXYGEN_GENERATE_TREEVIEW YES)
	set(DOXYGEN_DISABLE_INDEX NO)
	set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
	set(DOXYGEN_MARKDOWN_SUPPORT YES)
	set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${EYE_AI_CORE_DIR}/../README.md")
	set(DOXYGEN_LAYOUT_FILE "${EYE_AI_CORE_DIR}/docs/DoxygenLayout.xml")
	set(DOXYGEN_HTML_EXTRA_STYLESHEET
		"${AWESOME_CSS_DIR}/doxygen-awesome.css"
		"${AWESOME_CSS_DIR}/doxygen-awesome-sidebar-only.css"
		"${AWESOME_CSS_DIR}/doxygen-awesome-sidebar-only-darkmode-toggle.css"
		"${EYE_AI_CORE_DIR}/docs/custom.css"
	)
	set(DOXYGEN_HTML_HEADER "${EYE_AI_CORE_DIR}/docs/header.html")
	set(DOXYGEN_HTML_EXTRA_FILES
		"${AWESOME_CSS_DIR}/doxygen-awesome-darkmode-toggle.js"
		"${AWESOME_CSS_DIR}/doxygen-awesome-fragment-copy-button.js"
		${EYE_AI_FAVICON_PATH}
	)
	set(DOXYGEN_HTML_COPY_CLIPBOARD NO)
	set(DOXYGEN_HAVE_DOT YES)
	set(DOXYGEN_DOT_IMAGE_FORMAT svg)
	set(DOXYGEN_DOT_TRANSPARENT YES)
	set(DOXYGEN_TOC_INCLUDE_HEADINGS 0) # disables including of headings
	set(DOXYGEN_INTERACTIVE_SVG YES)

	doxygen_add_docs(
		EyeAICoreDoxygenDocs
		"${EYE_AI_CORE_DIR}/../README.md"
		"${EYE_AI_CORE_DIR}/../Beschreibung.md"
		"${EYE_AI_CORE_DIR}/README.md"
		"${EYE_AI_CORE_DIR}/docs"
		"${EYE_AI_CORE_DIR}/include/EyeAICore"
		"${EYE_AI_CORE_DIR}/src"
		"${EYE_AI_CORE_DIR}/metric_depth/README.md"
		"${EYE_AI_CORE_DIR}/../EyeAIApp/README.md"
		"${EYE_AI_CORE_DIR}/../EyeAIServer/README.md"
		"${EYE_AI_CORE_DIR}/../EyeAIServer/Client/README.md"
		"${EYE_AI_CORE_DIR}/../EyeAIServer/Server/README.md"
		"${EYE_AI_CORE_DIR}/../EyeAIApp/app/src/main/assets/README.md"
		COMMENT "Generating EyeAICore documentation with Doxygen"
	)
else()
	message(WARNING "Did not find Doxygen installation, not generating documentation even though EYE_AI_CORE_GENERATE_DOCS is enabled")
endif()