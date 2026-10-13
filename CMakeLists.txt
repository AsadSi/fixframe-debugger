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

set(UI_FILES ${CMAKE_SOURCE_DIR}/ui/index.html ${CMAKE_SOURCE_DIR}/ui/style.css ${CMAKE_SOURCE_DIR}/ui/app.js)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/gen/ui_assets.h
  COMMAND ${CMAKE_COMMAND} -DOUT=${CMAKE_BINARY_DIR}/gen/ui_assets.h "-DFILES=${UI_FILES}" -P ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
  DEPENDS ${UI_FILES} ${CMAKE_SOURCE_DIR}/cmake/embed.cmake
  VERBATIM)

find_package(Threads REQUIRED)
add_library(VkLayer_fixframe SHARED layer/layer.cpp core/core.cpp ${CMAKE_BINARY_DIR}/gen/ui_assets.h)
target_include_directories(VkLayer_fixframe PRIVATE third_party/vkroots third_party/vku third_party/httplib
                           third_party/stb ${CMAKE_BINARY_DIR}/gen)
target_link_libraries(VkLayer_fixframe PRIVATE Vulkan::Headers Threads::Threads)
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

# tests/smoke.cpp: the layer in a small Vulkan program, no game needed (native architecture only: it links the
# system's Vulkan loader). Run: VK_LAYER_PATH=build VK_INSTANCE_LAYERS=VK_LAYER_FIXFRAME_debug build/fixframe_smoke
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
  add_executable(fixframe_smoke tests/smoke.cpp)
  target_link_libraries(fixframe_smoke PRIVATE Vulkan::Vulkan)
  if(APPLE)
    set_target_properties(fixframe_smoke PROPERTIES OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR})
  endif()
endif()
