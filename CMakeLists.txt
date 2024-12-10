cmake_minimum_required(VERSION 3.24)
project(PointCloudDisplayer)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RELEASE")
endif()

set(CMAKE_DEBUG_POSTFIX d)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(lux-cmake-toolset CONFIG REQUIRED)
find_package(lux-cxx                  REQUIRED  COMPONENTS compile_time static_reflection)
find_package(lux-engine-core          REQUIRED  COMPONENTS math)
find_package(lux-engine-tools         REQUIRED  COMPONENTS ui)

find_package(Boost					  REQUIRED  COMPONENTS interprocess)

# library for communication between the displayer and the point cloud generator
generate_visibility_header(
    ENABLE_MACRO_NAME   LUX_POINT_CLOUD_DISPLAYER_LIBRARY
    PUBLIC_MACRO_NAME   LUX_POINT_CLOUD_DISPLAYER_PUBLIC
    GENERATE_FILE_PATH  visibility.h
)

add_component(
    COMPONENT_NAME pcd
    STATIC
    NAMESPACE      pcd
    SOURCE_FILES   ${CMAKE_CURRENT_SOURCE_DIR}/src/pcd/pcd.cpp
)

component_include_directories(
    pcd
    BUILD_TIME_EXPORT  
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${LUX_GENERATE_HEADER_DIR}
    INSTALL_TIME
        include
)

target_compile_definitions(
	pcd
	PRIVATE
		LUX_POINT_CLOUD_DISPLAYER_LIBRARY
)

target_link_libraries(
	pcd
	PRIVATE
		Boost::interprocess
)

# Main displayer executable
component_get_cmake_scripts(lux::engine::function::render LUX_RENDER_TOOLS)
foreach(tool ${LUX_RENDER_TOOLS})
	message("Including render tool: ${tool}")
	include(${tool})
endforeach()

set(SHADER_COMPILE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/gen/shaders)
set(SHADER_CONFIG_OUT_PATH     ${CMAKE_BINARY_DIR}/gen/include)

compile_glsl_shaders(
	OUTPUT_DIR		${SHADER_COMPILE_OUTPUT_PATH} 
	OUTPUT_SPIR_V	VERTEX_SHADER_PATH
	SHADERS			${CMAKE_CURRENT_SOURCE_DIR}/shaders/point_cloud.vert
)
compile_glsl_shaders(
	OUTPUT_DIR		${SHADER_COMPILE_OUTPUT_PATH} 
	OUTPUT_SPIR_V	FRAGMENT_SHADER_PATH
	SHADERS			${CMAKE_CURRENT_SOURCE_DIR}/shaders/point_cloud.frag
)

configure_file(
	${CMAKE_CURRENT_SOURCE_DIR}/shader_config.hpp.in 
	${SHADER_CONFIG_OUT_PATH}/shader_config.hpp
	@ONLY
)

add_executable(
    point_cloud_displayer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/displayer/entry.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/src/displayer/PointCloudScene.cpp
)

target_link_libraries(
	point_cloud_displayer
	PRIVATE
		lux::engine::tools::ui
		lux::cxx::static_reflection
		pcd
)

target_include_directories(
	point_cloud_displayer
	PRIVATE
		include
		pinclude
		${SHADER_CONFIG_OUT_PATH}
)

install_components(
	PROJECT_NAME	pcd
	VERSION			0.0.1
	NAMESPACE		pcd
	COMPONENTS		pcd
)
