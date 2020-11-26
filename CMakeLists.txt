cmake_minimum_required (VERSION 3.12)
project (StableNN)

#---------------------------------------------------------------------------------------
# main project
#---------------------------------------------------------------------------------------

# find all source and header files
file(GLOB_RECURSE SOURCE_FILES "src/*.cpp" "src/*.hpp" "src/*.inl")

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

add_executable(pendulum ${SOURCE_FILES})
set_target_properties(pendulum PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
)
if (MSVC)
	set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT pendulum)
endif (MSVC)
#---------------------------------------------------------------------------------------
# libraries
#---------------------------------------------------------------------------------------

find_package(Torch REQUIRED)
target_link_libraries(pendulum "${TORCH_LIBRARIES}")

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
	target_include_directories(pendulum PRIVATE "dependencies/SFML/include")
	add_subdirectory("dependencies/SFML")
endif()
target_link_libraries(pendulum sfml-system sfml-graphics sfml-window)
