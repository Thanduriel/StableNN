cmake_minimum_required (VERSION 3.12)
project (StableNN)

OPTION(USE_GRAPHICS "link SFML for visualisations" OFF)

if(CMAKE_CONFIGURATION_TYPES)      
	set(CMAKE_CONFIGURATION_TYPES Debug Release)
endif()

#---------------------------------------------------------------------------------------
# main project
#---------------------------------------------------------------------------------------

# find all source and header files
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.hpp" "src/*.inl")
list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/mainpendulum.cpp")
list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/pendulumeval.cpp")
list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/pendulumeval.hpp")
list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/mainheateq.cpp")
list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/heateqeval.hpp")

if(NOT USE_GRAPHICS)
	list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/src/evaluation/renderer.cpp")
endif(NOT USE_GRAPHICS)

# organize the files according to the organisation in the file system
# from: http:#cmake.3232098.n2.nabble.com/Keep-folder-structure-in-Visual-Studio-Project-generated-with-CMake-td7586044.html
macro(GroupSources curdir)
   file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir} ${PROJECT_SOURCE_DIR}/${curdir}/*)
   foreach(child ${children})
          if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
                  GroupSources(${curdir}/${child})
          else()
                  string(REPLACE "/" "\\" groupname ${curdir})
            source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
          endif()
   endforeach()
endmacro()

# execute tha macro
GroupSources(src)

add_library(stableNN STATIC ${SOURCE_FILES})
set_target_properties(stableNN PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)

add_executable(pendulum "src/mainpendulum.cpp" "src/pendulumeval.cpp" "src/pendulumeval.hpp")
set_target_properties(pendulum PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
target_link_libraries(pendulum stableNN)

add_executable(heateq "src/mainheateq.cpp" "src/heateqeval.hpp")
set_target_properties(heateq PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
target_link_libraries(heateq stableNN)

# enable link time code gen if supported
include(CheckIPOSupported)
check_ipo_supported(RESULT result)
if(result)
	set_property(TARGET stableNN pendulum heateq PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT pendulum)
else()
	# public options are propagated to the other targets
	target_compile_options(stableNN PUBLIC "$<$<CONFIG:DEBUG>:-Wall;-pedantic>")
	target_compile_options(stableNN PUBLIC "$<$<CONFIG:RELEASE>:-Wall;-pedantic;-O3;-march=native;-DNDEBUG>")
endif (MSVC)

#---------------------------------------------------------------------------------------
# libraries
#---------------------------------------------------------------------------------------

find_package(Torch REQUIRED)
target_link_libraries(stableNN PUBLIC "${TORCH_LIBRARIES}")

if (USE_GRAPHICS)
	find_package(SFML COMPONENTS system graphics window QUIET)
	if (NOT SFML_FOUND)
		message("SFML not found. Attempting to use local version.")
		set(SFML_BUILD_GRAPHICS TRUE)
		set(SFML_BUILD_WINDOW TRUE)
		set(BUILD_SHARED_LIBS TRUE)
		set(SFML_BUILD_AUDIO FALSE)
		set(SFML_BUILD_EXAMPLES FALSE)
		set(SFML_BUILD_DOC FALSE)
		set(SFML_BUILD_NETWORK FALSE)
		target_include_directories(stableNN PRIVATE "dependencies/SFML/include")
		add_subdirectory("dependencies/SFML")
	endif(NOT SFML_FOUND)
	target_link_libraries(stableNN PRIVATE sfml-system sfml-graphics sfml-window)
endif(USE_GRAPHICS)