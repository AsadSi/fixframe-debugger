// Copyright (c) 2026 asadsi. MIT licence (see LICENSE).
//
// Fixframe's SDK, for renderers a Vulkan layer can't see (Metal, a custom backend). The renderer reports what
// it draws while a capture is being recorded; Fixframe serves it in the same UI as the layer's captures.
//
// Use, from the render thread:
//   fixframe::sdk::start("My game", "Apple M4", "Metal");             once, when the renderer is up
//   if (fixframe::sdk::capturing()) { ... add passes/textures/draws } per draw: cheap when not capturing
//   fixframe::sdk::frame_end(frame, draws, [] { return png_of_tv(); }); once per frame, after its last draw
//
// Report every draw the game asked for, including the ones the renderer skipped (Draw::skipped says why):
// missing renders are often exactly those.
#pragma once
#include <functional>
#include <string>

#include "../core/core.h"

namespace fixframe::sdk {

void start(const std::string& app, const std::string& device, const std::string& backend);

// True while the current frame is captured; capture() is then the capture to add to.
bool capturing();
Capture* capture();

// Adds a pass if its targets differ from the last one's and returns its index (draws refer to it).
int pass(const Pass& p);

// The end of a frame: `draws` drawn in it; image() gives the presented picture as a PNG (core::png) and is
// called only when a capture needs it.
void frame_end(uint64_t frame, uint32_t draws, const std::function<std::string()>& image);

}  // namespace fixframe::sdk
