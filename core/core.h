// Copyright (c) 2026 asadsi. MIT licence (see LICENSE).
//
// Fixframe's core: the backend-neutral capture model and the local HTTP server that serves it. Backends (the
// Vulkan layer, the SDK) build Captures from what they record and publish them here; the UI and the CLI read
// them over HTTP. Core never touches a graphics API.
#pragma once
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fixframe {

struct KV {
    std::string key, value;
};

struct Shader {
    std::string stage, name;  // "vertex", "fs_0x00000000fbe7c4b6"
    uint64_t hash = 0;        // of the shader's code
    uint32_t size = 0;        // bytes
};

// What a backend draws with: a Vulkan pipeline, a GX2 shader pair with its state, ...
struct Program {
    uint64_t id = 0;
    std::string name;
    std::vector<Shader> shaders;
    std::vector<KV> state;  // fixed-function state, in a stable order so two programs diff by key
};

struct Texture {
    uint64_t id = 0;
    std::string name, format;
    uint32_t width = 0, height = 0, mips = 1, layers = 1;
};

// Where draws go: a render pass / dynamic rendering scope.
struct Pass {
    std::string label;
    std::vector<uint64_t> colors;  // Texture ids
    uint64_t depth = 0;
    uint32_t width = 0, height = 0;
};

struct Draw {
    std::string kind;  // draw, indexed, indirect, indexed-indirect, indirect-count, ..., dispatch
    uint32_t count = 0, instances = 0;
    int pass = -1;  // index in Capture::passes, -1 outside a pass (dispatches)
    std::string label;
    std::shared_ptr<const Program> program;
    std::vector<KV> dynamic;   // viewport, scissor, push constants: state set per draw
    std::vector<KV> textures;  // slot ("set 0 binding 3") -> Texture id (decimal)
};

struct Capture {
    uint32_t id = 0;
    uint64_t frame = 0;
    std::vector<Pass> passes;
    std::vector<Draw> draws;
    std::map<uint64_t, Texture> textures;
    std::string png;  // the presented image, empty if it could not be read back
};

namespace core {

// Starts the server once (later calls do nothing). app/device/backend describe the process for the UI.
void start(const std::string& app, const std::string& device, const std::string& backend);
// Backends, once a frame: the frame just presented and the draws it submitted.
void frame_done(uint64_t frame, uint32_t draws);
// Backends, at a frame boundary: true once per capture asked for (the UI's Capture button).
bool take_capture_request();
// Backends: a finished capture; core numbers it and keeps the newest few.
void publish(std::unique_ptr<Capture> capture);

}  // namespace core
}  // namespace fixframe
