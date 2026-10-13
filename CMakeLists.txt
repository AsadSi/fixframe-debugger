cmake_minimum_required(VERSION 3.24)
# macOS: one universal layer, so Rosetta (x86_64) and native (arm64) processes both load it.
if(APPLE AND NOT CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "")
endif()
project(fixframe CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(VulkanHeaders CONFIG QUIET)
if(NOT VulkanHeaders_FOUND)
  find_path(VULKAN_INCLUDE vulkan/vulkan.h REQUIRED)
  add_library(Vulkan::Headers INTERFACE IMPORTED)
  target_include_directories(Vulkan::Headers INTERFACE ${VULKAN_INCLUDE})
endif()

add_library(VkLayer_fixframe SHARED layer/layer.cpp)
target_include_directories(VkLayer_fixframe PRIVATE third_party/vkroots)
target_link_libraries(VkLayer_fixframe PRIVATE Vulkan::Headers)
set_target_properties(VkLayer_fixframe PROPERTIES CXX_VISIBILITY_PRESET hidden)
if(APPLE)
  # the loader looks for the name in the manifest, not a versioned one
  set_target_properties(VkLayer_fixframe PROPERTIES SUFFIX ".dylib")
endif()

if(APPLE)
  set(MANIFEST_LIB "./libVkLayer_fixframe.dylib")
else()
  set(MANIFEST_LIB "./libVkLayer_fixframe.so")
endif()
file(READ layer/VkLayer_fixframe.json MANIFEST)
string(REPLACE "./libVkLayer_fixframe.dylib" "${MANIFEST_LIB}" MANIFEST "${MANIFEST}")
file(WRITE ${CMAKE_BINARY_DIR}/VkLayer_fixframe.json "${MANIFEST}")
