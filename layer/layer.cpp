// Copyright (c) 2026 asadsi. MIT licence (see LICENSE).
//
// VK_LAYER_FIXFRAME_debug: records what each command buffer draws with (pipeline, textures, targets, labels,
// viewport, push constants) and, when the UI asks for a capture, turns the next frame's submits into a
// fixframe::Capture with the presented image. Recording is always on and cheap (a few small structs a draw);
// resolving handles into names and formats happens only for a captured frame, at submit time, while every
// object the frame uses is still alive. One device is assumed (emulators make one).
#include "vkroots.h"
#include "vk_enum_string_helper.h"

#include <atomic>
#include <cerrno>  // program_invocation_short_name (glibc)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <shared_mutex>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include "../core/core.h"


namespace fixframe {
namespace {

const char* arch() {
#if defined(__x86_64__)
    return "x86_64";
#elif defined(__aarch64__)
    return "arm64";
#else
    return "?";
#endif
}

const char* process_name() {
#if defined(__APPLE__)
    return getprogname();
#else
    return program_invocation_short_name;
#endif
}

std::string strip(const char* s, std::string_view prefix) {
    std::string_view v = s;
    if (v.starts_with(prefix)) v.remove_prefix(prefix.size());
    return std::string(v);
}

uint64_t fnv(const void* data, size_t size) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < size; i++) h = (h ^ static_cast<const uint8_t*>(data)[i]) * 0x100000001b3ull;
    return h;
}

template <typename T>
const T* find_in_chain(const void* next, VkStructureType type) {
    for (auto* s = static_cast<const VkBaseInStructure*>(next); s; s = s->pNext)
        if (s->sType == type) return reinterpret_cast<const T*>(s);
    return nullptr;
}

// ---------------------------------------------------------------------------------------------------------
// What the app made. Guarded by g_objects; looked up at bind time (pipelines) and at capture time (the rest).

struct ImageInfo {
    VkFormat format;
    VkExtent3D extent;
    uint32_t mips, layers;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    uint64_t serial = 0;  // unique per creation: the id captures use (the handle can be reused)
};
std::atomic<uint64_t> g_serial{1};

// What a descriptor slot holds (images only: buffers are not shown yet).
struct Slot {
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    bool operator==(const Slot&) const = default;
};

struct RenderPassInfo {
    std::vector<VkImageLayout> final_layout;  // per attachment: its layout after the pass
    struct Subpass {
        std::vector<uint32_t> colors;          // attachment indices (VK_ATTACHMENT_UNUSED kept: slot numbers)
        uint32_t depth = VK_ATTACHMENT_UNUSED;
    };
    std::vector<Subpass> subpasses;
};

struct Template {
    std::vector<VkDescriptorUpdateTemplateEntry> entries;
    VkPipelineBindPoint bind_point = VK_PIPELINE_BIND_POINT_GRAPHICS;  // push templates
};
struct SwapchainInfo {
    VkFormat format;
    VkExtent2D extent;
    bool readable;  // made with TRANSFER_SRC (we add it when the surface allows)
    std::vector<VkImage> images;
};

// Dynamic state the layer follows, as bits; a pipeline's mask says which of them it takes from the command
// buffer instead of its own create info. The keys match pipeline_state()'s, so a draw's values replace them.
enum DynBit : uint32_t { kCull = 1, kFront = 2, kTopology = 4, kDepthTest = 8, kDepthWrite = 16, kDepthCompare = 32,
                         kDepthBias = 64, kStencil = 128, kScissor = 256, kScissorCount = 512, kStencilCompareMask = 1024,
                         kStencilWriteMask = 2048, kStencilRef = 4096, kStencilOp = 8192, kBlendConstants = 16384,
                         kDepthBiasValues = 32768 };
struct Dyn {
    VkCullModeFlags cull = 0;
    VkFrontFace front = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCompareOp compare = VK_COMPARE_OP_NEVER;
    VkBool32 depth_test = 0, depth_write = 0, depth_bias = 0, stencil = 0;
    VkStencilOpState face[2]{};  // front, back: ops, compare op, masks, reference as set dynamically
    float blend[4]{}, bias[3]{};  // blend constants; depth bias constant, clamp, slope
};

struct PipelineRec {
    std::shared_ptr<const Program> program;  // replaced, not changed, when named: captures hold the old one
    uint32_t dynamic = 0;
};

std::shared_mutex g_objects;
std::unordered_map<uint64_t, std::string> g_names;  // any object, from vkSetDebugUtilsObjectNameEXT
std::unordered_map<VkShaderModule, Shader> g_modules;
std::unordered_map<VkPipeline, PipelineRec> g_pipelines;
std::unordered_map<VkImage, ImageInfo> g_images;
std::unordered_map<VkImageView, VkImage> g_views;
std::unordered_map<VkFramebuffer, std::vector<VkImageView>> g_framebuffers;
std::unordered_map<VkDescriptorSet, std::map<uint64_t, Slot>> g_sets;  // (binding << 32 | element) -> what it holds
std::unordered_map<VkDescriptorPool, std::unordered_set<VkDescriptorSet>> g_pool_sets;
std::unordered_map<VkDescriptorUpdateTemplate, Template> g_templates;
std::unordered_map<VkRenderPass, std::shared_ptr<const RenderPassInfo>> g_render_passes;
std::unordered_map<VkSampler, std::string> g_samplers;  // summary, as the inspector shows it
std::unordered_map<VkSwapchainKHR, SwapchainInfo> g_swapchains;

std::string name_of(uint64_t handle) {
    auto it = g_names.find(handle);
    return it == g_names.end() ? std::string() : it->second;
}

bool is_depth(VkFormat f) {
    switch (f) {
        case VK_FORMAT_D16_UNORM: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_S8_UINT:
        case VK_FORMAT_D16_UNORM_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
            return true;
        default:
            return false;
    }
}

// Descriptors whose VkDescriptorImageInfo the layer keeps: images and samplers (alone or combined).
bool is_image_descriptor(VkDescriptorType t) {
    return t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
           t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT || t == VK_DESCRIPTOR_TYPE_SAMPLER;
}

const char* stage_name(VkShaderStageFlagBits s) {
    switch (s) {
        case VK_SHADER_STAGE_VERTEX_BIT: return "vertex";
        case VK_SHADER_STAGE_FRAGMENT_BIT: return "fragment";
        case VK_SHADER_STAGE_GEOMETRY_BIT: return "geometry";
        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT: return "tess control";
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: return "tess eval";
        case VK_SHADER_STAGE_COMPUTE_BIT: return "compute";
        default: return "other";
    }
}

std::string blend_text(const VkPipelineColorBlendAttachmentState& a) {
    if (!a.blendEnable) return "off";
    auto f = [](VkBlendFactor x) { return strip(string_VkBlendFactor(x), "VK_BLEND_FACTOR_"); };
    auto op = [](VkBlendOp x) { return strip(string_VkBlendOp(x), "VK_BLEND_OP_"); };
    return "color " + f(a.srcColorBlendFactor) + " " + op(a.colorBlendOp) + " " + f(a.dstColorBlendFactor) +
           ", alpha " + f(a.srcAlphaBlendFactor) + " " + op(a.alphaBlendOp) + " " + f(a.dstAlphaBlendFactor);
}

std::string cull_text(VkCullModeFlags c) {
    if (!c) return "NONE";
    std::string s = strip(string_VkCullModeFlags(c).c_str(), "VK_CULL_MODE_");
    return s.ends_with("_BIT") ? s.substr(0, s.size() - 4) : s;
}
std::string on(VkBool32 b) { return b ? "on" : "off"; }

uint32_t dynamic_mask(const VkPipelineDynamicStateCreateInfo* dy) {
    uint32_t m = 0;
    for (uint32_t i = 0; dy && i < dy->dynamicStateCount; i++) switch (dy->pDynamicStates[i]) {
        case VK_DYNAMIC_STATE_CULL_MODE: m |= kCull; break;
        case VK_DYNAMIC_STATE_FRONT_FACE: m |= kFront; break;
        case VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY: m |= kTopology; break;
        case VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE: m |= kDepthTest; break;
        case VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE: m |= kDepthWrite; break;
        case VK_DYNAMIC_STATE_DEPTH_COMPARE_OP: m |= kDepthCompare; break;
        case VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE: m |= kDepthBias; break;
        case VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE: m |= kStencil; break;
        case VK_DYNAMIC_STATE_SCISSOR: m |= kScissor; break;
        case VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT: m |= kScissorCount; break;
        case VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK: m |= kStencilCompareMask; break;
        case VK_DYNAMIC_STATE_STENCIL_WRITE_MASK: m |= kStencilWriteMask; break;
        case VK_DYNAMIC_STATE_STENCIL_REFERENCE: m |= kStencilRef; break;
        case VK_DYNAMIC_STATE_STENCIL_OP: m |= kStencilOp; break;
        case VK_DYNAMIC_STATE_BLEND_CONSTANTS: m |= kBlendConstants; break;
        case VK_DYNAMIC_STATE_DEPTH_BIAS: m |= kDepthBiasValues; break;
        default: break;
    }
    return m;
}

std::string stencil_ops(const VkStencilOpState& f) {
    auto op = [](VkStencilOp o) { return strip(string_VkStencilOp(o), "VK_STENCIL_OP_"); };
    return strip(string_VkCompareOp(f.compareOp), "VK_COMPARE_OP_") + ", fail " + op(f.failOp) + ", pass " + op(f.passOp) +
           ", depth fail " + op(f.depthFailOp);
}
std::string pair_hex(uint32_t front, uint32_t back) {
    char b[48];
    snprintf(b, sizeof b, front == back ? "%X" : "front %X, back %X", front, back);
    return b;
}
std::string floats(const float* v, int n) {
    std::string s;
    char b[24];
    for (int i = 0; i < n; i++) snprintf(b, sizeof b, i ? " %g" : "%g", v[i]), s += b;
    return s;
}
std::string bias_text(const float* v) {
    char b[96];
    snprintf(b, sizeof b, "constant %g, clamp %g, slope %g", v[0], v[1], v[2]);
    return b;
}

void dynamic_state(std::vector<KV>& out, uint32_t mask, const Dyn& d) {
    if (mask & kTopology) out.push_back({"topology", strip(string_VkPrimitiveTopology(d.topology), "VK_PRIMITIVE_TOPOLOGY_")});
    if (mask & kCull) out.push_back({"cull", cull_text(d.cull)});
    if (mask & kFront) out.push_back({"front face", strip(string_VkFrontFace(d.front), "VK_FRONT_FACE_")});
    if (mask & kDepthBias) out.push_back({"depth bias", on(d.depth_bias)});
    if (mask & kDepthTest) out.push_back({"depth test", on(d.depth_test)});
    if (mask & kDepthWrite) out.push_back({"depth write", on(d.depth_write)});
    if (mask & kDepthCompare) out.push_back({"depth compare", strip(string_VkCompareOp(d.compare), "VK_COMPARE_OP_")});
    if (mask & kStencil) out.push_back({"stencil test", on(d.stencil)});
    if (mask & kStencilOp) {
        out.push_back({"stencil front", stencil_ops(d.face[0])});
        out.push_back({"stencil back", stencil_ops(d.face[1])});
    }
    if (mask & kStencilCompareMask) out.push_back({"stencil compare mask", pair_hex(d.face[0].compareMask, d.face[1].compareMask)});
    if (mask & kStencilWriteMask) out.push_back({"stencil write mask", pair_hex(d.face[0].writeMask, d.face[1].writeMask)});
    if (mask & kStencilRef) out.push_back({"stencil reference", pair_hex(d.face[0].reference, d.face[1].reference)});
    if (mask & kBlendConstants) out.push_back({"blend constants", floats(d.blend, 4)});
    if (mask & kDepthBiasValues) out.push_back({"depth bias values", bias_text(d.bias)});
}

std::string mask_text(VkColorComponentFlags m) {
    std::string s;
    if (m & VK_COLOR_COMPONENT_R_BIT) s += 'R';
    if (m & VK_COLOR_COMPONENT_G_BIT) s += 'G';
    if (m & VK_COLOR_COMPONENT_B_BIT) s += 'B';
    if (m & VK_COLOR_COMPONENT_A_BIT) s += 'A';
    return s.empty() ? "none" : s;
}

// Fixed-function state of a graphics pipeline, in a stable key order so two programs diff key by key.
std::vector<KV> pipeline_state(const VkGraphicsPipelineCreateInfo& ci) {
    std::vector<KV> s;
    if (auto* ia = ci.pInputAssemblyState)
        s.push_back({"topology", strip(string_VkPrimitiveTopology(ia->topology), "VK_PRIMITIVE_TOPOLOGY_")});
    if (auto* r = ci.pRasterizationState) {
        s.push_back({"cull", cull_text(r->cullMode)});
        s.push_back({"front face", strip(string_VkFrontFace(r->frontFace), "VK_FRONT_FACE_")});
        s.push_back({"polygon mode", strip(string_VkPolygonMode(r->polygonMode), "VK_POLYGON_MODE_")});
        s.push_back({"depth bias", on(r->depthBiasEnable)});
        float bias[3] = {r->depthBiasConstantFactor, r->depthBiasClamp, r->depthBiasSlopeFactor};
        s.push_back({"depth bias values", bias_text(bias)});
        s.push_back({"depth clamp", on(r->depthClampEnable)});
        s.push_back({"rasterizer discard", on(r->rasterizerDiscardEnable)});
    }
    if (auto* ms = ci.pMultisampleState) {
        std::string samples = strip(string_VkSampleCountFlagBits(ms->rasterizationSamples), "VK_SAMPLE_COUNT_");
        s.push_back({"samples", samples.substr(0, samples.find('_'))});
        s.push_back({"alpha to coverage", on(ms->alphaToCoverageEnable)});
    }
    if (auto* ds = ci.pDepthStencilState) {
        s.push_back({"depth test", on(ds->depthTestEnable)});
        s.push_back({"depth write", on(ds->depthWriteEnable)});
        s.push_back({"depth compare", strip(string_VkCompareOp(ds->depthCompareOp), "VK_COMPARE_OP_")});
        s.push_back({"stencil test", on(ds->stencilTestEnable)});
        s.push_back({"stencil front", stencil_ops(ds->front)});
        s.push_back({"stencil back", stencil_ops(ds->back)});
        s.push_back({"stencil compare mask", pair_hex(ds->front.compareMask, ds->back.compareMask)});
        s.push_back({"stencil write mask", pair_hex(ds->front.writeMask, ds->back.writeMask)});
        s.push_back({"stencil reference", pair_hex(ds->front.reference, ds->back.reference)});
    }
    if (auto* cb = ci.pColorBlendState) {
        for (uint32_t i = 0; i < cb->attachmentCount; i++) {
            s.push_back({"blend " + std::to_string(i), blend_text(cb->pAttachments[i])});
            s.push_back({"write mask " + std::to_string(i), mask_text(cb->pAttachments[i].colorWriteMask)});
        }
        if (cb->logicOpEnable) s.push_back({"logic op", strip(string_VkLogicOp(cb->logicOp), "VK_LOGIC_OP_")});
        s.push_back({"blend constants", floats(cb->blendConstants, 4)});
    }
    if (auto* dy = ci.pDynamicState) {
        std::string list;
        for (uint32_t i = 0; i < dy->dynamicStateCount; i++)
            list += (i ? ", " : "") + strip(string_VkDynamicState(dy->pDynamicStates[i]), "VK_DYNAMIC_STATE_");
        s.push_back({"dynamic", list});
    }
    return s;
}

Shader shader_of(const VkPipelineShaderStageCreateInfo& st) {
    Shader sh;
    if (st.module) {
        auto it = g_modules.find(st.module);
        if (it != g_modules.end()) sh = it->second;
        if (sh.name.empty()) sh.name = name_of((uint64_t)st.module);
    } else if (auto* mci = find_in_chain<VkShaderModuleCreateInfo>(st.pNext, VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO)) {
        sh.hash = fnv(mci->pCode, mci->codeSize), sh.size = (uint32_t)mci->codeSize;  // maintenance5: no module
    }
    sh.stage = stage_name(st.stage);
    return sh;
}

// ---------------------------------------------------------------------------------------------------------
// Per command buffer recording (in the command buffer's vkroots user data; one recording thread at a time).

enum Kind : uint8_t { kDraw, kIndexed, kIndirect, kIndexedIndirect, kIndirectCount, kIndexedIndirectCount, kDispatch,
                      kDispatchIndirect, kExecute };
const char* kKindNames[] = {"draw", "indexed", "indirect", "indexed-indirect", "indirect-count",
                            "indexed-indirect-count", "dispatch", "dispatch-indirect", "execute"};

struct Bindings {  // descriptor state of one bind point
    VkDescriptorSet sets[8] = {};
    std::map<uint64_t, Slot> pushed;  // (set << 48 | binding << 32 | element) -> what it holds
    VkPipelineLayout push_layout[8] = {};  // pushes stay until overwritten or pushed with another layout
};

struct PassRec {
    std::vector<VkImageView> views;  // colours then depth (told apart by format at capture)
    VkExtent2D extent;
    uint32_t label;
    // what a scrub copies when the pass ends: colour attachment 0, else depth (dynamic rendering only)
    VkImage image = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
    ImageInfo info{};
};

struct Rec {
    Kind kind;
    uint32_t a = 0, b = 0;  // vertices/indices/groups x, instances/groups y
    VkPipeline pipeline = VK_NULL_HANDLE;
    int pass = -1, bindings = -1, push = -1;
    uint32_t label = 0;
    VkViewport viewport{};
    VkRect2D scissor{};
    Dyn dyn;
    VkCommandBuffer secondary = VK_NULL_HANDLE;  // kExecute
    uint64_t key = 0;                            // Draw::key
    VkQueryPool pool = VK_NULL_HANDLE;           // pick frames: its occlusion query
    uint32_t query = 0;
    bool outside = false;                        // pick frames: the pixel is outside its viewport or scissor
    bool approximate = false;                    // pick frames: counted over the whole draw (static scissor)
};

struct CbState {
    std::vector<Rec> recs;
    std::vector<PassRec> passes;
    std::vector<Bindings> bindings;
    std::vector<std::vector<uint8_t>> pushes;
    uint32_t draws = 0;
    // current state
    VkPipeline pipeline[2] = {};  // graphics, compute
    Bindings cur[2];
    int bound[2] = {-1, -1};  // index in bindings, -1: changed since last recorded
    std::vector<uint8_t> push_data;
    int push = -1;
    int pass = -1;
    std::vector<uint32_t> labels;
    VkViewport viewport{};
    VkRect2D scissor{};
    bool scissor_ext = false;  // the app sets scissors through the EXT entry point
    Dyn dyn;
    // a job's frame (see Job)
    const void* job = nullptr;   // the pick job its query pool belongs to
    VkQueryPool pool = VK_NULL_HANDLE;
    uint32_t queries = 0;
    bool in_query = false;
    int clipped = 0;             // the draw's scissor was set to the picked pixel: 1 SetScissor, 2 WithCount
    int scrub_pass = -1;         // the pass of the scrubbed draw: its later draws are skipped
    // the render pass object being recorded (its subpasses are pass records from pass_scope on)
    std::shared_ptr<const RenderPassInfo> render_pass;
    std::vector<VkImageView> rp_views;
    VkExtent2D rp_extent{};
    uint32_t subpass = 0;
    int pass_scope = -1;

    void reset() { *this = CbState(); }
};

CbState& state(const vkroots::VkCommandBufferDispatch& d) {
    if (!d.UserData.has()) d.UserData.emplace<CbState>();
    return d.UserData.cast<CbState>();
}

int point(VkPipelineBindPoint bp) { return bp == VK_PIPELINE_BIND_POINT_COMPUTE ? 1 : 0; }

// Debug label paths ("Frame / Shadows / Draw"), interned for the life of the process.
std::mutex g_label_mutex;
std::vector<std::string> g_labels{""};
std::unordered_map<std::string, uint32_t> g_label_ids{{"", 0}};

uint32_t intern(const std::string& s) {
    std::lock_guard lock(g_label_mutex);
    auto [it, added] = g_label_ids.emplace(s, (uint32_t)g_labels.size());
    if (added) g_labels.push_back(s);
    return it->second;
}
std::string label_text(uint32_t id) {
    std::lock_guard lock(g_label_mutex);
    return g_labels[id];
}

Rec& record(CbState& s, Kind kind, uint32_t a, uint32_t b) {
    int bp = kind == kDispatch || kind == kDispatchIndirect ? 1 : 0;
    if (s.bound[bp] < 0) {
        s.bindings.push_back(s.cur[bp]);
        s.bound[bp] = (int)s.bindings.size() - 1;
    }
    if (s.push < 0 && !s.push_data.empty()) {
        s.pushes.push_back(s.push_data);
        s.push = (int)s.pushes.size() - 1;
    }
    Rec r{kind, a, b, s.pipeline[bp], bp ? -1 : s.pass, s.bound[bp], s.push, s.labels.empty() ? 0 : s.labels.back(),
          s.viewport, s.scissor, s.dyn};
    uint64_t parts[] = {(uint64_t)r.pipeline, r.kind, r.a, r.b, r.label};
    r.key = fnv(parts, sizeof parts);
    s.recs.push_back(r);
    s.draws++;
    return s.recs.back();
}

// The slots a push writes, for one set of one bind point (after `start_push`).
Bindings& start_push(CbState& s, VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set) {
    Bindings& b = s.cur[point(bp)];
    if (b.push_layout[set] != layout)  // another layout: what was pushed before is gone
        std::erase_if(b.pushed, [&](auto& p) { return (p.first >> 48) == set; });
    b.push_layout[set] = layout;
    b.sets[set] = VK_NULL_HANDLE;
    s.bound[point(bp)] = -1;
    return b;
}

void push_descriptors(const vkroots::VkCommandBufferDispatch& d, VkPipelineBindPoint bp, VkPipelineLayout layout,
                      uint32_t set, uint32_t count, const VkWriteDescriptorSet* writes) {
    if (set >= 8) return;
    Bindings& b = start_push(state(d), bp, layout, set);
    for (uint32_t i = 0; i < count; i++) {
        const VkWriteDescriptorSet& w = writes[i];
        if (!is_image_descriptor(w.descriptorType) || !w.pImageInfo) continue;
        for (uint32_t e = 0; e < w.descriptorCount; e++)
            b.pushed[(uint64_t)set << 48 | (uint64_t)w.dstBinding << 32 | (w.dstArrayElement + e)] =
                {w.pImageInfo[e].imageView, w.pImageInfo[e].sampler};
    }
}

// The image slots a template writes, from the app's data (update or push).
template <typename F>
void template_slots(const Template& t, const void* data, F&& write) {
    for (const VkDescriptorUpdateTemplateEntry& e : t.entries) {
        if (!is_image_descriptor(e.descriptorType)) continue;
        for (uint32_t i = 0; i < e.descriptorCount; i++) {
            auto* info = reinterpret_cast<const VkDescriptorImageInfo*>((const char*)data + e.offset + e.stride * i);
            write((uint64_t)e.dstBinding << 32 | (e.dstArrayElement + i), Slot{info->imageView, info->sampler});
        }
    }
}

void bind_sets(CbState& s, VkPipelineBindPoint bp, uint32_t first, uint32_t count, const VkDescriptorSet* sets) {
    Bindings& b = s.cur[point(bp)];
    for (uint32_t i = 0; i < count && first + i < 8; i++) {
        b.sets[first + i] = sets[i];
        std::erase_if(b.pushed, [&](auto& p) { return (p.first >> 48) == first + i; });
        b.push_layout[first + i] = VK_NULL_HANDLE;
    }
    s.bound[point(bp)] = -1;
}

// Starts a pass record; `target` (colour 0, else depth) in `layout` once the pass is over is what a scrub copies.
void begin_pass(const vkroots::VkCommandBufferDispatch& d, std::vector<VkImageView> views, VkExtent2D extent,
                VkImageView target, VkImageLayout layout, bool depth) {
    CbState& s = state(d);
    s.passes.push_back({std::move(views), extent, s.labels.empty() ? 0 : s.labels.back()});
    s.pass = (int)s.passes.size() - 1;
    if (!target) return;
    PassRec& p = s.passes.back();
    std::shared_lock lock(g_objects);
    auto v = g_views.find(target);
    auto i = v == g_views.end() ? g_images.end() : g_images.find(v->second);
    if (i == g_images.end()) return;
    p.image = i->first, p.layout = layout, p.info = i->second;
    p.aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
}

void begin_rendering(const vkroots::VkCommandBufferDispatch& d, const VkRenderingInfo* info) {
    std::vector<VkImageView> views;
    for (uint32_t i = 0; i < info->colorAttachmentCount; i++)
        if (info->pColorAttachments[i].imageView) views.push_back(info->pColorAttachments[i].imageView);
    if (info->pDepthAttachment && info->pDepthAttachment->imageView) views.push_back(info->pDepthAttachment->imageView);
    else if (info->pStencilAttachment && info->pStencilAttachment->imageView)
        views.push_back(info->pStencilAttachment->imageView);
    const VkRenderingAttachmentInfo* a = info->colorAttachmentCount && info->pColorAttachments[0].imageView
                                             ? &info->pColorAttachments[0] : info->pDepthAttachment;
    CbState& s = state(d);
    s.pass_scope = (int)s.passes.size();
    begin_pass(d, std::move(views), info->renderArea.extent, a ? a->imageView : VK_NULL_HANDLE,
               a ? a->imageLayout : VK_IMAGE_LAYOUT_UNDEFINED, a && a == info->pDepthAttachment);
}

// A subpass of a render pass object: its own colour and depth attachments, as a pass record of its own.
void begin_subpass(const vkroots::VkCommandBufferDispatch& d) {
    CbState& s = state(d);
    const RenderPassInfo* rp = s.render_pass.get();
    if (!rp || s.subpass >= rp->subpasses.size()) {  // not known: every attachment, no scrub target
        begin_pass(d, s.rp_views, s.rp_extent, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED, false);
        return;
    }
    const RenderPassInfo::Subpass& sp = rp->subpasses[s.subpass];
    std::vector<VkImageView> views;
    for (uint32_t a : sp.colors)
        if (a < s.rp_views.size()) views.push_back(s.rp_views[a]);
    if (sp.depth < s.rp_views.size()) views.push_back(s.rp_views[sp.depth]);
    uint32_t t = !sp.colors.empty() && sp.colors[0] < s.rp_views.size() ? sp.colors[0] : sp.depth;
    bool ok = t < s.rp_views.size() && t < rp->final_layout.size();
    begin_pass(d, std::move(views), s.rp_extent, ok ? s.rp_views[t] : VK_NULL_HANDLE,
               ok ? rp->final_layout[t] : VK_IMAGE_LAYOUT_UNDEFINED, ok && t == sp.depth);
}

void begin_render_pass(const vkroots::VkCommandBufferDispatch& d, const VkRenderPassBeginInfo* info) {
    CbState& s = state(d);
    s.rp_views.clear();
    if (auto* imageless = find_in_chain<VkRenderPassAttachmentBeginInfo>(info->pNext,
                                                                         VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO)) {
        s.rp_views.assign(imageless->pAttachments, imageless->pAttachments + imageless->attachmentCount);
    }
    {
        std::shared_lock lock(g_objects);
        if (s.rp_views.empty())
            if (auto it = g_framebuffers.find(info->framebuffer); it != g_framebuffers.end()) s.rp_views = it->second;
        auto rp = g_render_passes.find(info->renderPass);
        s.render_pass = rp == g_render_passes.end() ? nullptr : rp->second;
    }
    s.rp_extent = info->renderArea.extent;
    s.subpass = 0;
    s.pass_scope = (int)s.passes.size();
    begin_subpass(d);
}

void next_subpass(const vkroots::VkCommandBufferDispatch& d) {
    CbState& s = state(d);
    s.subpass++;
    begin_subpass(d);
}

// ---------------------------------------------------------------------------------------------------------
// Frames, captures and one-frame jobs (pick, scrub)

struct DeviceInfo {
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    std::unordered_map<VkQueue, uint32_t> queue_family;
};
DeviceInfo g_device;

std::atomic<uint64_t> g_frame{0};
std::atomic<uint32_t> g_frame_draws{0};
std::mutex g_capture_mutex;              // the capture being built (submits can come from several threads)
std::unique_ptr<Capture> g_building;     // non-null while the frame being captured is submitted
std::vector<std::pair<VkQueryPool, uint32_t>> g_pick_slots;  // pick captures: each draw's query

struct Readback;

// A one-frame experiment the UI asked for. Games record a frame while the one before is still being presented, so a
// job armed at a present stays on for two: the frame submitted between the second and third present was recorded
// wholly inside it (that frame is the pick capture), and its results are read at the third. Its Vulkan objects
// live a few frames longer (retired), in case a command buffer that used them is submitted again without being
// recorded again.
struct Job {
    Request req;
    std::mutex m;
    std::vector<VkQueryPool> pools;  // kPick: one per primary command buffer
    Readback* target = nullptr;  // kScrub
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    bool found = false, copied = false;
    std::string error;
    int presents = 0;  // seen since it was armed
    bool counting = false;  // kScrub: the anchor was recorded; draws with the target's key since it
    uint32_t counted = 0;
    uint64_t retire_at = 0;
};
std::atomic<Job*> g_job{nullptr};   // the job of the frame being recorded
std::unique_ptr<Job> g_running;     // its owner (present thread)
std::vector<std::unique_ptr<Job>> g_retired;
constexpr uint32_t kQueries = 4096;

// A host-visible buffer an image is copied into.
struct Readback {
    const vkroots::VkDeviceDispatch* d = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;

    bool create(const vkroots::VkDeviceDispatch& dd, VkDeviceSize size) {
        d = &dd;
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = size;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (d->CreateBuffer(d->Device, &bci, nullptr, &buffer) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        d->GetBufferMemoryRequirements(d->Device, buffer, &req);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < g_device.memory.memoryTypeCount; i++)
            if ((req.memoryTypeBits & (1u << i)) && (g_device.memory.memoryTypes[i].propertyFlags & want) == want) {
                VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, i};
                return d->AllocateMemory(d->Device, &mai, nullptr, &memory) == VK_SUCCESS &&
                       d->BindBufferMemory(d->Device, buffer, memory, 0) == VK_SUCCESS;
            }
        return false;
    }
    const uint8_t* map() {
        void* p = nullptr;
        return d->MapMemory(d->Device, memory, 0, VK_WHOLE_SIZE, 0, &p) == VK_SUCCESS ? (const uint8_t*)p : nullptr;
    }
    void destroy() {  // freeing mapped memory unmaps it
        if (buffer) d->DestroyBuffer(d->Device, buffer, nullptr);
        if (memory) d->FreeMemory(d->Device, memory, nullptr);
        buffer = VK_NULL_HANDLE, memory = VK_NULL_HANDLE;
    }
};

// Records a copy of mip 0, layer 0 of an image into a buffer; the image is in `layout` before and after.
void record_copy(const vkroots::VkDeviceDispatch& d, VkCommandBuffer cb, VkImage image, VkImageAspectFlags aspect,
                 VkImageLayout layout, VkExtent3D extent, VkBuffer buffer) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = layout;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {aspect, 0, 0, 1};
    copy.imageExtent = {extent.width, extent.height, 1};
    d.CmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
    std::swap(b.oldLayout, b.newLayout);
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
}

// --- pictures of images

float half(uint16_t h) {
    int e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? std::ldexp((float)m, -24) : e == 31 ? 65504.f : std::ldexp((float)(m | 1024), e - 25);
    return h & 0x8000 ? -v : v;
}
float small_float(uint32_t bits, int mantissa) {  // the unsigned 11- and 10-bit floats of B10G11R11
    uint32_t e = bits >> mantissa, m = bits & ((1u << mantissa) - 1);
    if (e == 0) return std::ldexp((float)m, -14 - mantissa);
    if (e == 31) return 65504.f;
    return std::ldexp((float)(m | (1u << mantissa)), (int)e - 15 - mantissa);
}

// Bytes per texel of the formats the layer can show (depth formats: their depth aspect); 0 for the others.
uint32_t texel_size(VkFormat f) {
    switch (f) {
        case VK_FORMAT_R8_UNORM: return 1;
        case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_D16_UNORM: return 2;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM:
        case VK_FORMAT_B8G8R8A8_SRGB: case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R32_SFLOAT:
        case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
            return 4;
        case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
        case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
        default: return 0;
    }
}

// A PNG of an image's texels. Float colour (scene buffers, linear) is clamped to 0..1 and gamma-encoded; depth is
// stretched over its own range so it can be seen. Empty for formats texel_size() does not know.
std::string picture(VkFormat f, const uint8_t* src, uint32_t w, uint32_t h) {
    uint32_t ts = texel_size(f);
    if (!ts) return {};
    size_t n = size_t(w) * h;
    std::vector<float> rgb(n * 3);
    bool linear = false;
    for (size_t i = 0; i < n; i++) {
        const uint8_t* p = src + i * ts;
        float* o = &rgb[i * 3];
        uint32_t v = 0;
        memcpy(&v, p, std::min<uint32_t>(ts, 4));
        float x;
        switch (f) {
            case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
                o[0] = p[0] / 255.f, o[1] = p[1] / 255.f, o[2] = p[2] / 255.f;
                break;
            case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
                o[0] = p[2] / 255.f, o[1] = p[1] / 255.f, o[2] = p[0] / 255.f;
                break;
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
                o[0] = (v & 1023) / 1023.f, o[1] = ((v >> 10) & 1023) / 1023.f, o[2] = ((v >> 20) & 1023) / 1023.f;
                break;
            case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
                o[2] = (v & 1023) / 1023.f, o[1] = ((v >> 10) & 1023) / 1023.f, o[0] = ((v >> 20) & 1023) / 1023.f;
                break;
            case VK_FORMAT_R8_UNORM: o[0] = o[1] = o[2] = p[0] / 255.f; break;
            case VK_FORMAT_R8G8_UNORM: o[0] = p[0] / 255.f, o[1] = p[1] / 255.f, o[2] = 0; break;
            case VK_FORMAT_R16_SFLOAT: o[0] = o[1] = o[2] = half(v & 0xffff), linear = true; break;
            case VK_FORMAT_R16G16_SFLOAT: o[0] = half(v & 0xffff), o[1] = half(v >> 16), o[2] = 0, linear = true; break;
            case VK_FORMAT_R16G16B16A16_SFLOAT: {
                uint16_t c[3];
                memcpy(c, p, 6);
                o[0] = half(c[0]), o[1] = half(c[1]), o[2] = half(c[2]), linear = true;
                break;
            }
            case VK_FORMAT_R32G32B32A32_SFLOAT: memcpy(o, p, 12), linear = true; break;
            case VK_FORMAT_B10G11R11_UFLOAT_PACK32:
                o[0] = small_float(v & 2047, 6), o[1] = small_float((v >> 11) & 2047, 6), o[2] = small_float(v >> 22, 5);
                linear = true;
                break;
            case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_D32_SFLOAT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
                memcpy(&x, p, 4), o[0] = o[1] = o[2] = x;
                break;
            case VK_FORMAT_D16_UNORM: o[0] = o[1] = o[2] = (v & 0xffff) / 65535.f; break;
            default: o[0] = o[1] = o[2] = (v & 0xffffff) / 16777215.f; break;  // D24
        }
    }
    if (is_depth(f)) {  // stretch: depth sits in a narrow band near one end
        float lo = INFINITY, hi = -INFINITY;
        for (size_t i = 0; i < n; i++) lo = std::min(lo, rgb[i * 3]), hi = std::max(hi, rgb[i * 3]);
        float scale = hi > lo ? 1 / (hi - lo) : 0;
        for (float& c : rgb) c = (c - lo) * scale;
    }
    std::vector<uint8_t> rgba(n * 4, 255);
    for (size_t i = 0; i < n; i++)
        for (int c = 0; c < 3; c++) {
            float x = std::clamp(rgb[i * 3 + c], 0.f, 1.f);
            if (linear) x = std::pow(x, 1 / 2.2f);
            rgba[i * 4 + c] = uint8_t(x * 255 + 0.5f);
        }
    return core::png(rgba.data(), w, h);
}

// --- recording, resolving

std::string hex(const std::vector<uint8_t>& v) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    size_t n = std::min<size_t>(v.size(), 128);
    for (size_t i = 0; i < n; i++) {
        if (i && i % 4 == 0) s += ' ';
        s += digits[v[i] >> 4], s += digits[v[i] & 15];
    }
    if (n < v.size()) s += " …";
    return s;
}

// Before each draw or dispatch: records it and applies the job of the frame, if any. False: skip the command.
//   pick: the draw is clipped to the picked pixel (where its pipeline takes the scissor dynamically) and counted by
//         an occlusion query, so the query says whether it wrote that pixel;
//   scrub: the draws after the scrubbed one, in its pass, are skipped.
bool begin_draw(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, Kind kind, uint32_t a, uint32_t b) {
    CbState& s = state(d);
    Rec& r = record(s, kind, a, b);
    Job* job = g_job.load(std::memory_order_acquire);
    if (!job) return true;
    if (job->req.kind == Request::kScrub) {
        if (kind < kDispatch && s.scrub_pass >= 0 && s.pass == s.scrub_pass) return false;
        std::lock_guard lock(job->m);
        if (job->found) return true;  // the job spans two frames: the first match only
        bool hit = false;
        if (r.key == job->req.anchor) {
            job->counting = true, job->counted = 0;
            hit = job->req.anchor == job->req.target;
        } else if (job->counting && r.key == job->req.target) {
            hit = job->counted++ == job->req.nth;
        }
        if (hit) {
            job->found = true;
            if (kind < kDispatch && s.pass >= 0) s.scrub_pass = s.pass;
        }
        return true;
    }
    if (kind >= kDispatch) return true;
    if (s.job != job || s.queries >= kQueries) return true;  // begun before the job: no query pool
    uint32_t dyn = 0;
    {
        std::shared_lock lock(g_objects);
        if (auto it = g_pipelines.find(r.pipeline); it != g_pipelines.end()) dyn = it->second.dynamic;
    }
    // The picked pixel, 0..1 of the screen, in this draw's framebuffer: the viewport rectangle covers the screen
    // top-down whatever the sign of its height (a negative height flips the picture, not the rectangle).
    VkViewport vp = s.viewport;
    if (vp.width == 0 && s.pass >= 0)
        vp = {0, 0, (float)s.passes[s.pass].extent.width, (float)s.passes[s.pass].extent.height, 0, 1};
    float top = std::min(vp.y, vp.y + vp.height);
    int32_t x = (int32_t)std::floor(vp.x + job->req.x * vp.width);
    int32_t y = (int32_t)std::floor(top + job->req.y * std::fabs(vp.height));
    if (dyn & (kScissor | kScissorCount)) {
        const VkRect2D& sc = s.scissor;
        bool inside = x >= std::max(sc.offset.x, 0) && y >= std::max(sc.offset.y, 0) &&
                      x < sc.offset.x + (int32_t)sc.extent.width && y < sc.offset.y + (int32_t)sc.extent.height;
        if (!inside) {  // it can't write the pixel: no query needed
            r.outside = true;
            return true;
        }
        VkRect2D px{{x, y}, {1, 1}};
        if (dyn & kScissorCount) {
            if (s.scissor_ext) d.CmdSetScissorWithCountEXT(cb, 1, &px);
            else d.CmdSetScissorWithCount(cb, 1, &px);
            s.clipped = 2;
        } else {
            d.CmdSetScissor(cb, 0, 1, &px);
            s.clipped = 1;
        }
    } else {
        r.approximate = true;  // a static scissor can't be narrowed without a pipeline variant (M4)
    }
    r.pool = s.pool, r.query = s.queries++;
    d.CmdBeginQuery(cb, r.pool, r.query, 0);
    s.in_query = true;
    return true;
}

void end_draw(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb) {
    CbState& s = state(d);
    if (!s.in_query) return;
    d.CmdEndQuery(cb, s.recs.back().pool, s.recs.back().query);
    if (s.clipped == 2) {
        if (s.scissor_ext) d.CmdSetScissorWithCountEXT(cb, 1, &s.scissor);
        else d.CmdSetScissorWithCount(cb, 1, &s.scissor);
    } else if (s.clipped == 1) {
        d.CmdSetScissor(cb, 0, 1, &s.scissor);
    }
    s.in_query = false, s.clipped = 0;
}

// After a pass ends: in a scrub frame, the scrubbed draw's target is copied out here (outside the pass).
void end_pass(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb) {
    CbState& s = state(d);
    int first = s.pass_scope, last = s.pass;  // the pass records this end closes (render pass: all its subpasses)
    s.pass = -1, s.pass_scope = -1, s.render_pass = nullptr;
    Job* job = g_job.load(std::memory_order_acquire);
    if (!job || s.scrub_pass < 0 || s.scrub_pass < first || s.scrub_pass > last) return;
    int pass = s.scrub_pass;
    s.scrub_pass = -1;
    const PassRec& p = s.passes[pass];
    std::lock_guard lock(job->m);
    if (job->copied) return;
    job->copied = true;
    uint32_t ts = texel_size(p.info.format);
    if (!p.image) job->error = "Its target is not known (its render pass or framebuffer was not seen being made).";
    else if (p.info.samples != VK_SAMPLE_COUNT_1_BIT) job->error = "Its target is multisampled; Fixframe can't show those yet.";
    else if (!ts) job->error = "Fixframe can't show its target's format (" + strip(string_VkFormat(p.info.format), "VK_FORMAT_") + ") yet.";
    if (!job->error.empty()) return;
    auto* rb = new Readback;
    if (!rb->create(*d.pDeviceDispatch, VkDeviceSize(ts) * p.info.extent.width * p.info.extent.height)) {
        rb->destroy();
        delete rb;
        job->error = "No host-visible memory for the copy.";
        return;
    }
    record_copy(*d.pDeviceDispatch, cb, p.image, p.aspect, p.layout, p.info.extent, rb->buffer);
    job->target = rb, job->format = p.info.format, job->extent = {p.info.extent.width, p.info.extent.height};
}

uint64_t add_texture(Capture& c, VkImageView view) {  // g_objects held
    auto v = g_views.find(view);
    if (v == g_views.end()) return 0;
    VkImage image = v->second;
    auto it = g_images.find(image);
    if (it == g_images.end()) return 0;
    uint64_t key = it->second.serial;
    if (!c.textures.count(key)) {
        Texture t;
        t.id = key;
        t.name = name_of((uint64_t)image);
        t.format = strip(string_VkFormat(it->second.format), "VK_FORMAT_");
        t.width = it->second.extent.width, t.height = it->second.extent.height;
        t.mips = it->second.mips, t.layers = it->second.layers;
        c.textures[key] = t;
    }
    return key;
}

bool view_is_depth(VkImageView view) {  // g_objects held
    auto v = g_views.find(view);
    if (v == g_views.end()) return false;
    auto i = g_images.find(v->second);
    return i != g_images.end() && is_depth(i->second.format);
}

// Appends one submitted command buffer's records to the capture. inherit_pass: the primary's pass for a
// secondary's draws. g_objects held (shared), g_capture_mutex held.
void resolve(Capture& c, VkCommandBuffer cb, int inherit_pass) {
    auto* d = vkroots::LookupDispatch(cb);
    if (!d || !d->UserData.has()) return;
    CbState& s = d->UserData.cast<CbState>();
    int base = (int)c.passes.size();
    for (const PassRec& p : s.passes) {
        Pass out;
        out.label = label_text(p.label);
        out.width = p.extent.width, out.height = p.extent.height;
        for (VkImageView v : p.views) {
            if (view_is_depth(v)) out.depth = add_texture(c, v);
            else if (uint64_t t = add_texture(c, v)) out.colors.push_back(t);
        }
        c.passes.push_back(std::move(out));
    }
    for (const Rec& r : s.recs) {
        int pass = r.pass >= 0 ? base + r.pass : inherit_pass;
        if (r.kind == kExecute) {
            resolve(c, r.secondary, pass);
            continue;
        }
        Draw out;
        out.kind = kKindNames[r.kind];
        out.key = r.key;
        out.count = r.a, out.instances = r.b;
        out.pass = r.kind >= kDispatch ? -1 : pass;
        out.label = label_text(r.label);
        if (auto it = g_pipelines.find(r.pipeline); it != g_pipelines.end()) {
            out.program = it->second.program;
            dynamic_state(out.dynamic, it->second.dynamic, r.dyn);
        }
        if (r.kind < kDispatch) {
            char b[96];
            snprintf(b, sizeof b, "%g %g %g x %g, depth %g..%g", r.viewport.x, r.viewport.y, r.viewport.width,
                     r.viewport.height, r.viewport.minDepth, r.viewport.maxDepth);
            out.dynamic.push_back({"viewport", b});
            snprintf(b, sizeof b, "%d %d %u x %u", r.scissor.offset.x, r.scissor.offset.y, r.scissor.extent.width,
                     r.scissor.extent.height);
            out.dynamic.push_back({"scissor", b});
        }
        if (r.push >= 0) out.dynamic.push_back({"push constants", hex(s.pushes[r.push])});
        if (r.bindings >= 0) {
            const Bindings& b = s.bindings[r.bindings];
            auto slot = [](uint64_t set, uint64_t binding, uint64_t element) {
                return "set " + std::to_string(set) + " binding " + std::to_string(binding) +
                       (element ? "[" + std::to_string(element) + "]" : "");
            };
            auto add = [&](std::string name, const Slot& sl) {
                if (uint64_t t = add_texture(c, sl.view)) out.textures.push_back({name, std::to_string(t)});
                if (auto it = sl.sampler ? g_samplers.find(sl.sampler) : g_samplers.end(); it != g_samplers.end())
                    out.samplers.push_back({name, it->second});
            };
            for (uint32_t set = 0; set < 8; set++) {
                auto it = b.sets[set] ? g_sets.find(b.sets[set]) : g_sets.end();
                if (it == g_sets.end()) continue;
                for (auto& [k, sl] : it->second) add(slot(set, k >> 32, k & 0xffffffff), sl);
            }
            for (auto& [k, sl] : b.pushed) add(slot(k >> 48, (k >> 32) & 0xffff, k & 0xffffffff), sl);
        }
        if (c.picked) {
            if (r.outside) out.samples = 0;
            out.approximate = r.approximate;
        }
        c.draws.push_back(std::move(out));
        if (c.picked) g_pick_slots.push_back({r.pool, r.query});
    }
}

uint32_t count_draws(VkCommandBuffer cb) {
    auto* d = vkroots::LookupDispatch(cb);
    if (!d || !d->UserData.has()) return 0;
    CbState& s = d->UserData.cast<CbState>();
    uint32_t n = s.draws;
    for (const Rec& r : s.recs)
        if (r.kind == kExecute) n += count_draws(r.secondary);
    return n;
}

void submitted(const VkCommandBuffer* cbs, uint32_t count) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < count; i++) n += count_draws(cbs[i]);
    g_frame_draws += n;
    std::lock_guard lock(g_capture_mutex);
    if (!g_building) return;
    std::shared_lock objects(g_objects);
    for (uint32_t i = 0; i < count; i++) resolve(*g_building, cbs[i], -1);
}

// Presents with the swapchain image copied to the host first; png gets the image (empty on failure). The copy is
// submitted on the present queue, waiting for what the present waited for; the present then waits for the copy.
VkResult present_with_readback(const vkroots::VkQueueDispatch& qd, VkQueue queue, const VkPresentInfoKHR* info,
                               std::string& png) {
    const vkroots::VkDeviceDispatch& d = *qd.pDeviceDispatch;
    VkDevice dev = d.Device;
    SwapchainInfo sc;
    {
        std::shared_lock lock(g_objects);
        auto it = g_swapchains.find(info->pSwapchains[0]);
        if (it == g_swapchains.end()) return qd.QueuePresentKHR(queue, info);
        sc = it->second;
    }
    uint32_t index = info->pImageIndices[0];
    auto family = g_device.queue_family.find(queue);
    if (!sc.readable || !texel_size(sc.format) || index >= sc.images.size() || family == g_device.queue_family.end())
        return qd.QueuePresentKHR(queue, info);

    Readback rb;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkSemaphore done = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    auto cleanup = [&] {
        if (fence) d.DestroyFence(dev, fence, nullptr);
        if (done) d.DestroySemaphore(dev, done, nullptr);
        if (pool) d.DestroyCommandPool(dev, pool, nullptr);
        rb.destroy();
    };
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, family->second};
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, VK_NULL_HANDLE,
                                    VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    if (!rb.create(d, VkDeviceSize(texel_size(sc.format)) * sc.extent.width * sc.extent.height) ||
        d.CreateCommandPool(dev, &pci, nullptr, &pool) != VK_SUCCESS ||
        d.CreateSemaphore(dev, &sci, nullptr, &done) != VK_SUCCESS || d.CreateFence(dev, &fci, nullptr, &fence) != VK_SUCCESS ||
        (cai.commandPool = pool, d.AllocateCommandBuffers(dev, &cai, &cb)) != VK_SUCCESS) {
        cleanup();
        return qd.QueuePresentKHR(queue, info);
    }
    *reinterpret_cast<void**>(cb) = *reinterpret_cast<void**>(dev);  // a layer's own dispatchable object: the loader's table

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                   VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    d.BeginCommandBuffer(cb, &begin);
    record_copy(d, cb, sc.images[index], VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                {sc.extent.width, sc.extent.height, 1}, rb.buffer);
    d.EndCommandBuffer(cb);

    std::vector<VkPipelineStageFlags> stages(info->waitSemaphoreCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = info->waitSemaphoreCount;
    submit.pWaitSemaphores = info->pWaitSemaphores;
    submit.pWaitDstStageMask = stages.data();
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &done;
    VkResult result;
    if (d.QueueSubmit(queue, 1, &submit, fence) != VK_SUCCESS) {
        result = qd.QueuePresentKHR(queue, info);
    } else {
        VkPresentInfoKHR waited = *info;
        waited.waitSemaphoreCount = 1;
        waited.pWaitSemaphores = &done;
        result = qd.QueuePresentKHR(queue, &waited);
        if (d.WaitForFences(dev, 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS)
            if (const uint8_t* p = rb.map()) png = picture(sc.format, p, sc.extent.width, sc.extent.height);
    }
    cleanup();
    return result;
}

void retire(const vkroots::VkDeviceDispatch& d, uint64_t frame) {  // present thread
    std::erase_if(g_retired, [&](std::unique_ptr<Job>& j) {
        if (j->retire_at > frame) return false;
        for (VkQueryPool p : j->pools) d.DestroyQueryPool(d.Device, p, nullptr);
        if (j->target) j->target->destroy(), delete j->target;
        return true;
    });
}

// ---------------------------------------------------------------------------------------------------------

// VK_EXT_debug_utils: object names and labels. The loader fetches these through vkGetInstanceProcAddr (an
// instance extension), which vkroots does not hook for device commands, so the layer answers for them itself
// (get_instance_proc_addr below) and calls the next layer through pointers taken from the instance chain.

PFN_vkSetDebugUtilsObjectNameEXT g_next_set_name;
PFN_vkCmdBeginDebugUtilsLabelEXT g_next_begin_label;
PFN_vkCmdEndDebugUtilsLabelEXT g_next_end_label;

void note_name(const VkDebugUtilsObjectNameInfoEXT* info) {
    if (!info->pObjectName) return;
    std::unique_lock lock(g_objects);
    g_names[info->objectHandle] = info->pObjectName;
    if (info->objectType == VK_OBJECT_TYPE_PIPELINE)
        if (auto it = g_pipelines.find((VkPipeline)info->objectHandle); it != g_pipelines.end()) {
            auto named = std::make_shared<Program>(*it->second.program);
            named->name = info->pObjectName;
            it->second.program = named;
        }
    if (info->objectType == VK_OBJECT_TYPE_SHADER_MODULE)
        if (auto it = g_modules.find((VkShaderModule)info->objectHandle); it != g_modules.end())
            it->second.name = info->pObjectName;
}

void begin_label(VkCommandBuffer cb, const VkDebugUtilsLabelEXT* label) {
    auto* d = vkroots::LookupDispatch(cb);
    if (!d || !label->pLabelName) return;
    CbState& s = state(*d);
    std::string parent = s.labels.empty() ? std::string() : label_text(s.labels.back());
    s.labels.push_back(intern(parent.empty() ? label->pLabelName : parent + " / " + label->pLabelName));
}

void end_label(VkCommandBuffer cb) {
    auto* d = vkroots::LookupDispatch(cb);
    if (d && !state(*d).labels.empty()) state(*d).labels.pop_back();
}

VKAPI_ATTR VkResult VKAPI_CALL hook_set_name(VkDevice dev, const VkDebugUtilsObjectNameInfoEXT* info) {
    note_name(info);
    return g_next_set_name(dev, info);
}
VKAPI_ATTR void VKAPI_CALL hook_begin_label(VkCommandBuffer cb, const VkDebugUtilsLabelEXT* label) {
    begin_label(cb, label);
    g_next_begin_label(cb, label);
}
VKAPI_ATTR void VKAPI_CALL hook_end_label(VkCommandBuffer cb) {
    end_label(cb);
    g_next_end_label(cb);
}

struct Instance {
    static VkResult CreateInstance(PFN_vkCreateInstance next, const VkInstanceCreateInfo* info,
                                   const VkAllocationCallbacks* alloc, VkInstance* instance) {
        const char* app = info->pApplicationInfo && info->pApplicationInfo->pApplicationName
                              ? info->pApplicationInfo->pApplicationName : "?";
        fprintf(stderr, "[fixframe] loaded (%s) into \"%s\"\n", arch(), app);
        VkResult r = next(info, alloc, instance);
        if (auto* id = r == VK_SUCCESS ? vkroots::LookupDispatch(*instance) : nullptr) {
            g_next_set_name = (PFN_vkSetDebugUtilsObjectNameEXT)id->GetInstanceProcAddr(*instance, "vkSetDebugUtilsObjectNameEXT");
            g_next_begin_label = (PFN_vkCmdBeginDebugUtilsLabelEXT)id->GetInstanceProcAddr(*instance, "vkCmdBeginDebugUtilsLabelEXT");
            g_next_end_label = (PFN_vkCmdEndDebugUtilsLabelEXT)id->GetInstanceProcAddr(*instance, "vkCmdEndDebugUtilsLabelEXT");
        }
        return r;
    }

    static VkResult CreateDevice(const vkroots::VkPhysicalDeviceDispatch& d, VkPhysicalDevice gpu,
                                 const VkDeviceCreateInfo* info, const VkAllocationCallbacks* alloc, VkDevice* device) {
        VkResult r = d.CreateDevice(gpu, info, alloc, device);
        if (r != VK_SUCCESS) return r;
        VkPhysicalDeviceProperties p;
        d.GetPhysicalDeviceProperties(gpu, &p);
        g_device.gpu = gpu;
        d.GetPhysicalDeviceMemoryProperties(gpu, &g_device.memory);
        char version[32];
        snprintf(version, sizeof version, "Vulkan %u.%u.%u", VK_API_VERSION_MAJOR(p.apiVersion),
                 VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        fprintf(stderr, "[fixframe] device: %s, %s\n", p.deviceName, version);
        if (auto* dd = vkroots::LookupDispatch(*device))  // the queues (vkroots made their tables already)
            for (uint32_t i = 0; i < info->queueCreateInfoCount; i++)
                for (uint32_t j = 0; j < info->pQueueCreateInfos[i].queueCount; j++) {
                    VkQueue q = VK_NULL_HANDLE;
                    dd->GetDeviceQueue(*device, info->pQueueCreateInfos[i].queueFamilyIndex, j, &q);
                    if (q) g_device.queue_family[q] = info->pQueueCreateInfos[i].queueFamilyIndex;
                }
        core::start(process_name(),std::string(p.deviceName) + ", " + version, std::string("Vulkan layer, ") + arch());
        return r;
    }
};

struct Device {
    // --- objects ---
    static VkResult SetDebugUtilsObjectNameEXT(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                               const VkDebugUtilsObjectNameInfoEXT* info) {
        note_name(info);
        return d.SetDebugUtilsObjectNameEXT(dev, info);
    }

    static VkResult CreateShaderModule(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                       const VkShaderModuleCreateInfo* info, const VkAllocationCallbacks* alloc,
                                       VkShaderModule* module) {
        VkResult r = d.CreateShaderModule(dev, info, alloc, module);
        if (r == VK_SUCCESS) {
            Shader s;
            s.hash = fnv(info->pCode, info->codeSize), s.size = (uint32_t)info->codeSize;
            std::unique_lock lock(g_objects);
            g_modules[*module] = s;
        }
        return r;
    }
    static void DestroyShaderModule(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkShaderModule module,
                                    const VkAllocationCallbacks* alloc) {
        {
            std::unique_lock lock(g_objects);
            g_modules.erase(module);
        }
        d.DestroyShaderModule(dev, module, alloc);
    }

    static VkResult CreateGraphicsPipelines(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkPipelineCache cache,
                                            uint32_t count, const VkGraphicsPipelineCreateInfo* infos,
                                            const VkAllocationCallbacks* alloc, VkPipeline* pipelines) {
        VkResult r = d.CreateGraphicsPipelines(dev, cache, count, infos, alloc, pipelines);
        std::unique_lock lock(g_objects);
        for (uint32_t i = 0; i < count; i++) {
            if (!pipelines[i]) continue;
            auto p = std::make_shared<Program>();
            p->id = (uint64_t)pipelines[i];
            for (uint32_t s = 0; s < infos[i].stageCount; s++) p->shaders.push_back(shader_of(infos[i].pStages[s]));
            p->state = pipeline_state(infos[i]);
            g_pipelines[pipelines[i]] = {p, dynamic_mask(infos[i].pDynamicState)};
        }
        return r;
    }
    static VkResult CreateComputePipelines(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkPipelineCache cache,
                                           uint32_t count, const VkComputePipelineCreateInfo* infos,
                                           const VkAllocationCallbacks* alloc, VkPipeline* pipelines) {
        VkResult r = d.CreateComputePipelines(dev, cache, count, infos, alloc, pipelines);
        std::unique_lock lock(g_objects);
        for (uint32_t i = 0; i < count; i++) {
            if (!pipelines[i]) continue;
            auto p = std::make_shared<Program>();
            p->id = (uint64_t)pipelines[i];
            p->shaders.push_back(shader_of(infos[i].stage));
            g_pipelines[pipelines[i]] = {p, 0};
        }
        return r;
    }
    static void DestroyPipeline(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkPipeline pipeline,
                                const VkAllocationCallbacks* alloc) {
        {
            std::unique_lock lock(g_objects);
            g_pipelines.erase(pipeline);  // captures keep their own reference
        }
        d.DestroyPipeline(dev, pipeline, alloc);
    }

    static VkResult CreateImage(const vkroots::VkDeviceDispatch& d, VkDevice dev, const VkImageCreateInfo* info,
                                const VkAllocationCallbacks* alloc, VkImage* image) {
        VkResult r = d.CreateImage(dev, info, alloc, image);
        if (r == VK_SUCCESS) {
            std::unique_lock lock(g_objects);
            g_images[*image] = {info->format, info->extent, info->mipLevels, info->arrayLayers, info->samples, g_serial++};
        }
        return r;
    }
    static void DestroyImage(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkImage image,
                             const VkAllocationCallbacks* alloc) {
        {
            std::unique_lock lock(g_objects);
            g_images.erase(image);
        }
        d.DestroyImage(dev, image, alloc);
    }
    static VkResult CreateImageView(const vkroots::VkDeviceDispatch& d, VkDevice dev, const VkImageViewCreateInfo* info,
                                    const VkAllocationCallbacks* alloc, VkImageView* view) {
        VkResult r = d.CreateImageView(dev, info, alloc, view);
        if (r == VK_SUCCESS) {
            std::unique_lock lock(g_objects);
            g_views[*view] = info->image;
        }
        return r;
    }
    static void DestroyImageView(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkImageView view,
                                 const VkAllocationCallbacks* alloc) {
        {
            std::unique_lock lock(g_objects);
            g_views.erase(view);
        }
        d.DestroyImageView(dev, view, alloc);
    }
    static VkResult CreateFramebuffer(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                      const VkFramebufferCreateInfo* info, const VkAllocationCallbacks* alloc,
                                      VkFramebuffer* fb) {
        VkResult r = d.CreateFramebuffer(dev, info, alloc, fb);
        if (r == VK_SUCCESS && info->pAttachments && !(info->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT)) {
            std::unique_lock lock(g_objects);
            g_framebuffers[*fb].assign(info->pAttachments, info->pAttachments + info->attachmentCount);
        }
        return r;
    }
    static void DestroyFramebuffer(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkFramebuffer fb,
                                   const VkAllocationCallbacks* alloc) {
        {
            std::unique_lock lock(g_objects);
            g_framebuffers.erase(fb);
        }
        d.DestroyFramebuffer(dev, fb, alloc);
    }

    static void UpdateDescriptorSets(const vkroots::VkDeviceDispatch& d, VkDevice dev, uint32_t writeCount,
                                     const VkWriteDescriptorSet* writes, uint32_t copyCount,
                                     const VkCopyDescriptorSet* copies) {
        {
            std::unique_lock lock(g_objects);
            for (uint32_t i = 0; i < writeCount; i++) {
                const VkWriteDescriptorSet& w = writes[i];
                if (!is_image_descriptor(w.descriptorType) || !w.pImageInfo) continue;
                auto& set = g_sets[w.dstSet];
                for (uint32_t e = 0; e < w.descriptorCount; e++)
                    set[(uint64_t)w.dstBinding << 32 | (w.dstArrayElement + e)] = {w.pImageInfo[e].imageView, w.pImageInfo[e].sampler};
            }
            for (uint32_t i = 0; i < copyCount; i++) {
                const VkCopyDescriptorSet& c = copies[i];
                auto src = g_sets.find(c.srcSet);
                if (src == g_sets.end()) continue;
                std::map<uint64_t, Slot> from = src->second;  // (dst may be src: copy first)
                auto& dst = g_sets[c.dstSet];
                for (uint32_t e = 0; e < c.descriptorCount; e++) {
                    uint64_t to = (uint64_t)c.dstBinding << 32 | (c.dstArrayElement + e);
                    auto it = from.find((uint64_t)c.srcBinding << 32 | (c.srcArrayElement + e));
                    if (it != from.end()) dst[to] = it->second;
                    else dst.erase(to);
                }
            }
        }
        d.UpdateDescriptorSets(dev, writeCount, writes, copyCount, copies);
    }

    // render passes: each subpass's attachments and every attachment's final layout
    template <typename Info>
    static void note_render_pass(VkRenderPass rp, const Info* info) {
        auto out = std::make_shared<RenderPassInfo>();
        for (uint32_t i = 0; i < info->attachmentCount; i++) out->final_layout.push_back(info->pAttachments[i].finalLayout);
        for (uint32_t i = 0; i < info->subpassCount; i++) {
            const auto& sp = info->pSubpasses[i];
            RenderPassInfo::Subpass o;
            for (uint32_t c = 0; c < sp.colorAttachmentCount; c++) o.colors.push_back(sp.pColorAttachments[c].attachment);
            if (sp.pDepthStencilAttachment) o.depth = sp.pDepthStencilAttachment->attachment;
            out->subpasses.push_back(std::move(o));
        }
        std::unique_lock lock(g_objects);
        g_render_passes[rp] = std::move(out);
    }
    static VkResult CreateRenderPass(const vkroots::VkDeviceDispatch& d, VkDevice dev, const VkRenderPassCreateInfo* info,
                                     const VkAllocationCallbacks* alloc, VkRenderPass* rp) {
        VkResult r = d.CreateRenderPass(dev, info, alloc, rp);
        if (r == VK_SUCCESS) note_render_pass(*rp, info);
        return r;
    }
    static VkResult CreateRenderPass2(const vkroots::VkDeviceDispatch& d, VkDevice dev, const VkRenderPassCreateInfo2* info,
                                      const VkAllocationCallbacks* alloc, VkRenderPass* rp) {
        VkResult r = d.CreateRenderPass2(dev, info, alloc, rp);
        if (r == VK_SUCCESS) note_render_pass(*rp, info);
        return r;
    }
    static VkResult CreateRenderPass2KHR(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                         const VkRenderPassCreateInfo2* info, const VkAllocationCallbacks* alloc,
                                         VkRenderPass* rp) {
        VkResult r = d.CreateRenderPass2KHR(dev, info, alloc, rp);
        if (r == VK_SUCCESS) note_render_pass(*rp, info);
        return r;
    }
    static void DestroyRenderPass(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkRenderPass rp,
                                  const VkAllocationCallbacks* alloc) {
        {
            std::unique_lock lock(g_objects);
            g_render_passes.erase(rp);  // command buffers being recorded keep their own reference
        }
        d.DestroyRenderPass(dev, rp, alloc);
    }

    static VkResult CreateSampler(const vkroots::VkDeviceDispatch& d, VkDevice dev, const VkSamplerCreateInfo* i,
                                  const VkAllocationCallbacks* alloc, VkSampler* sampler) {
        VkResult r = d.CreateSampler(dev, i, alloc, sampler);
        if (r != VK_SUCCESS) return r;
        auto f = [](VkFilter x) { return strip(string_VkFilter(x), "VK_FILTER_"); };
        auto a = [](VkSamplerAddressMode x) { return strip(string_VkSamplerAddressMode(x), "VK_SAMPLER_ADDRESS_MODE_"); };
        std::string wrap = i->addressModeU == i->addressModeV && i->addressModeV == i->addressModeW
                               ? a(i->addressModeU) : a(i->addressModeU) + "/" + a(i->addressModeV) + "/" + a(i->addressModeW);
        char b[200];
        snprintf(b, sizeof b, "%s/%s, mip %s, wrap %s, lod %g..%g bias %g%s", f(i->minFilter).c_str(), f(i->magFilter).c_str(),
                 strip(string_VkSamplerMipmapMode(i->mipmapMode), "VK_SAMPLER_MIPMAP_MODE_").c_str(), wrap.c_str(), i->minLod,
                 i->maxLod, i->mipLodBias, i->anisotropyEnable ? (", aniso " + std::to_string((int)i->maxAnisotropy)).c_str() : "");
        std::string text = b;
        if (i->compareEnable) text += ", compare " + strip(string_VkCompareOp(i->compareOp), "VK_COMPARE_OP_");
        std::unique_lock lock(g_objects);
        g_samplers[*sampler] = text;
        return r;
    }
    static void DestroySampler(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkSampler sampler,
                               const VkAllocationCallbacks* alloc) {
        {
            std::unique_lock lock(g_objects);
            g_samplers.erase(sampler);
        }
        d.DestroySampler(dev, sampler, alloc);
    }

    // descriptor sets live in pools: freed or reset with them (handles are reused; old contents must not show)
    static VkResult AllocateDescriptorSets(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                           const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* sets) {
        VkResult r = d.AllocateDescriptorSets(dev, info, sets);
        if (r == VK_SUCCESS) {
            std::unique_lock lock(g_objects);
            auto& pool = g_pool_sets[info->descriptorPool];
            for (uint32_t i = 0; i < info->descriptorSetCount; i++) g_sets.erase(sets[i]), pool.insert(sets[i]);
        }
        return r;
    }
    static VkResult FreeDescriptorSets(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkDescriptorPool pool,
                                       uint32_t count, const VkDescriptorSet* sets) {
        {
            std::unique_lock lock(g_objects);
            auto it = g_pool_sets.find(pool);
            for (uint32_t i = 0; i < count; i++) {
                g_sets.erase(sets[i]);
                if (it != g_pool_sets.end()) it->second.erase(sets[i]);
            }
        }
        return d.FreeDescriptorSets(dev, pool, count, sets);
    }
    static void forget_pool(VkDescriptorPool pool, bool destroyed) {
        std::unique_lock lock(g_objects);
        auto it = g_pool_sets.find(pool);
        if (it == g_pool_sets.end()) return;
        for (VkDescriptorSet s : it->second) g_sets.erase(s);
        if (destroyed) g_pool_sets.erase(it);
        else it->second.clear();
    }
    static VkResult ResetDescriptorPool(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkDescriptorPool pool,
                                        VkDescriptorPoolResetFlags flags) {
        forget_pool(pool, false);
        return d.ResetDescriptorPool(dev, pool, flags);
    }
    static void DestroyDescriptorPool(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkDescriptorPool pool,
                                      const VkAllocationCallbacks* alloc) {
        forget_pool(pool, true);
        d.DestroyDescriptorPool(dev, pool, alloc);
    }

    // update templates (emulators that update descriptors in bulk use them)
    static void note_template(VkDescriptorUpdateTemplate t, const VkDescriptorUpdateTemplateCreateInfo* info) {
        Template out;
        out.entries.assign(info->pDescriptorUpdateEntries, info->pDescriptorUpdateEntries + info->descriptorUpdateEntryCount);
        out.bind_point = info->pipelineBindPoint;
        std::unique_lock lock(g_objects);
        g_templates[t] = std::move(out);
    }
    static VkResult CreateDescriptorUpdateTemplate(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                                   const VkDescriptorUpdateTemplateCreateInfo* info,
                                                   const VkAllocationCallbacks* alloc, VkDescriptorUpdateTemplate* t) {
        VkResult r = d.CreateDescriptorUpdateTemplate(dev, info, alloc, t);
        if (r == VK_SUCCESS) note_template(*t, info);
        return r;
    }
    static VkResult CreateDescriptorUpdateTemplateKHR(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                                      const VkDescriptorUpdateTemplateCreateInfo* info,
                                                      const VkAllocationCallbacks* alloc, VkDescriptorUpdateTemplate* t) {
        VkResult r = d.CreateDescriptorUpdateTemplateKHR(dev, info, alloc, t);
        if (r == VK_SUCCESS) note_template(*t, info);
        return r;
    }
    static void forget_template(VkDescriptorUpdateTemplate t) {
        std::unique_lock lock(g_objects);
        g_templates.erase(t);
    }
    static void DestroyDescriptorUpdateTemplate(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                                VkDescriptorUpdateTemplate t, const VkAllocationCallbacks* alloc) {
        forget_template(t);
        d.DestroyDescriptorUpdateTemplate(dev, t, alloc);
    }
    static void DestroyDescriptorUpdateTemplateKHR(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                                   VkDescriptorUpdateTemplate t, const VkAllocationCallbacks* alloc) {
        forget_template(t);
        d.DestroyDescriptorUpdateTemplateKHR(dev, t, alloc);
    }
    static void update_with_template(VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void* data) {
        std::unique_lock lock(g_objects);
        auto it = g_templates.find(t);
        if (it == g_templates.end()) return;
        auto& slots = g_sets[set];
        template_slots(it->second, data, [&](uint64_t k, Slot sl) { slots[k] = sl; });
    }
    static void UpdateDescriptorSetWithTemplate(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkDescriptorSet set,
                                                VkDescriptorUpdateTemplate t, const void* data) {
        update_with_template(set, t, data);
        d.UpdateDescriptorSetWithTemplate(dev, set, t, data);
    }
    static void UpdateDescriptorSetWithTemplateKHR(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkDescriptorSet set,
                                                   VkDescriptorUpdateTemplate t, const void* data) {
        update_with_template(set, t, data);
        d.UpdateDescriptorSetWithTemplateKHR(dev, set, t, data);
    }

    static VkResult CreateSwapchainKHR(const vkroots::VkDeviceDispatch& d, VkDevice dev,
                                       const VkSwapchainCreateInfoKHR* info, const VkAllocationCallbacks* alloc,
                                       VkSwapchainKHR* swapchain) {
        // Copying the image out for the capture preview needs TRANSFER_SRC; add it where the surface allows.
        VkSurfaceCapabilitiesKHR caps{};
        d.pPhysicalDeviceDispatch->GetPhysicalDeviceSurfaceCapabilitiesKHR(d.PhysicalDevice, info->surface, &caps);
        VkSwapchainCreateInfoKHR ci = *info;
        bool readable = caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (readable) ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        VkResult r = d.CreateSwapchainKHR(dev, &ci, alloc, swapchain);
        if (r == VK_SUCCESS) {
            std::unique_lock lock(g_objects);
            g_swapchains[*swapchain] = {ci.imageFormat, ci.imageExtent, readable, {}};
        }
        return r;
    }
    static VkResult GetSwapchainImagesKHR(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkSwapchainKHR swapchain,
                                          uint32_t* count, VkImage* images) {
        VkResult r = d.GetSwapchainImagesKHR(dev, swapchain, count, images);
        if (images && (r == VK_SUCCESS || r == VK_INCOMPLETE)) {
            std::unique_lock lock(g_objects);
            auto& sc = g_swapchains[swapchain];
            sc.images.assign(images, images + *count);
            for (uint32_t i = 0; i < *count; i++)
                g_images[images[i]] = {sc.format, {sc.extent.width, sc.extent.height, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT, g_serial++};
        }
        return r;
    }
    static void DestroySwapchainKHR(const vkroots::VkDeviceDispatch& d, VkDevice dev, VkSwapchainKHR swapchain,
                                    const VkAllocationCallbacks* alloc) {
        {
            std::unique_lock lock(g_objects);
            if (auto it = g_swapchains.find(swapchain); it != g_swapchains.end()) {
                for (VkImage i : it->second.images) g_images.erase(i);
                g_swapchains.erase(it);
            }
        }
        d.DestroySwapchainKHR(dev, swapchain, alloc);
    }

    // --- recording ---
    static VkResult BeginCommandBuffer(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                       const VkCommandBufferBeginInfo* info) {
        state(d).reset();
        VkResult r = d.BeginCommandBuffer(cb, info);
        Job* job = g_job.load(std::memory_order_acquire);
        if (r == VK_SUCCESS && job && job->req.kind == Request::kPick && !info->pInheritanceInfo) {
            // a pick frame: this command buffer's draws are counted by queries of its own pool
            const vkroots::VkDeviceDispatch& dd = *d.pDeviceDispatch;
            VkQueryPoolCreateInfo qci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_OCCLUSION, kQueries};
            VkQueryPool pool;
            if (dd.CreateQueryPool(dd.Device, &qci, nullptr, &pool) == VK_SUCCESS) {
                d.CmdResetQueryPool(cb, pool, 0, kQueries);
                CbState& s = state(d);
                s.pool = pool, s.job = job;
                std::lock_guard lock(job->m);
                job->pools.push_back(pool);
            }
        }
        return r;
    }
    static VkResult ResetCommandBuffer(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                       VkCommandBufferResetFlags flags) {
        state(d).reset();
        return d.ResetCommandBuffer(cb, flags);
    }

    static void CmdBindPipeline(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkPipelineBindPoint bp,
                                VkPipeline pipeline) {
        state(d).pipeline[point(bp)] = pipeline;
        d.CmdBindPipeline(cb, bp, pipeline);
    }
    static void CmdBindDescriptorSets(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                      VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t first, uint32_t count,
                                      const VkDescriptorSet* sets, uint32_t dynCount, const uint32_t* dyn) {
        bind_sets(state(d), bp, first, count, sets);
        d.CmdBindDescriptorSets(cb, bp, layout, first, count, sets, dynCount, dyn);
    }
    // maintenance6 (Vulkan 1.4): one call can bind for graphics and compute
    static void bind_sets2(const vkroots::VkCommandBufferDispatch& d, const VkBindDescriptorSetsInfo* i) {
        if (i->stageFlags & ~VK_SHADER_STAGE_COMPUTE_BIT)
            bind_sets(state(d), VK_PIPELINE_BIND_POINT_GRAPHICS, i->firstSet, i->descriptorSetCount, i->pDescriptorSets);
        if (i->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
            bind_sets(state(d), VK_PIPELINE_BIND_POINT_COMPUTE, i->firstSet, i->descriptorSetCount, i->pDescriptorSets);
    }
    static void CmdBindDescriptorSets2(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                       const VkBindDescriptorSetsInfo* i) {
        bind_sets2(d, i);
        d.CmdBindDescriptorSets2(cb, i);
    }
    static void CmdBindDescriptorSets2KHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                          const VkBindDescriptorSetsInfo* i) {
        bind_sets2(d, i);
        d.CmdBindDescriptorSets2KHR(cb, i);
    }
    static void push_set2(const vkroots::VkCommandBufferDispatch& d, const VkPushDescriptorSetInfo* i) {
        if (i->stageFlags & ~VK_SHADER_STAGE_COMPUTE_BIT)
            push_descriptors(d, VK_PIPELINE_BIND_POINT_GRAPHICS, i->layout, i->set, i->descriptorWriteCount, i->pDescriptorWrites);
        if (i->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
            push_descriptors(d, VK_PIPELINE_BIND_POINT_COMPUTE, i->layout, i->set, i->descriptorWriteCount, i->pDescriptorWrites);
    }
    static void CmdPushDescriptorSet2(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                      const VkPushDescriptorSetInfo* i) {
        push_set2(d, i);
        d.CmdPushDescriptorSet2(cb, i);
    }
    static void CmdPushDescriptorSet2KHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                         const VkPushDescriptorSetInfo* i) {
        push_set2(d, i);
        d.CmdPushDescriptorSet2KHR(cb, i);
    }
    static void push_template(const vkroots::VkCommandBufferDispatch& d, VkDescriptorUpdateTemplate t,
                              VkPipelineLayout layout, uint32_t set, const void* data) {
        if (set >= 8) return;
        Template tmpl;
        {
            std::shared_lock lock(g_objects);
            auto it = g_templates.find(t);
            if (it == g_templates.end()) return;
            tmpl = it->second;
        }
        Bindings& b = start_push(state(d), tmpl.bind_point, layout, set);
        template_slots(tmpl, data, [&](uint64_t k, Slot sl) { b.pushed[(uint64_t)set << 48 | k] = sl; });
    }
    static void CmdPushDescriptorSetWithTemplate(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                                 VkDescriptorUpdateTemplate t, VkPipelineLayout layout, uint32_t set,
                                                 const void* data) {
        push_template(d, t, layout, set, data);
        d.CmdPushDescriptorSetWithTemplate(cb, t, layout, set, data);
    }
    static void CmdPushDescriptorSetWithTemplateKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                                    VkDescriptorUpdateTemplate t, VkPipelineLayout layout, uint32_t set,
                                                    const void* data) {
        push_template(d, t, layout, set, data);
        d.CmdPushDescriptorSetWithTemplateKHR(cb, t, layout, set, data);
    }
    static void CmdPushDescriptorSetKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                        VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t count,
                                        const VkWriteDescriptorSet* writes) {
        push_descriptors(d, bp, layout, set, count, writes);
        d.CmdPushDescriptorSetKHR(cb, bp, layout, set, count, writes);
    }
    static void CmdPushDescriptorSet(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                     VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t count,
                                     const VkWriteDescriptorSet* writes) {
        push_descriptors(d, bp, layout, set, count, writes);
        d.CmdPushDescriptorSet(cb, bp, layout, set, count, writes);
    }
    static void CmdPushConstants(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkPipelineLayout layout,
                                 VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* values) {
        push_constants(state(d), offset, size, values);
        d.CmdPushConstants(cb, layout, stages, offset, size, values);
    }
    static void push_constants(CbState& s, uint32_t offset, uint32_t size, const void* values) {
        if (s.push_data.size() < offset + size) s.push_data.resize(offset + size);
        memcpy(s.push_data.data() + offset, values, size);
        s.push = -1;
    }
    static void CmdPushConstants2(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, const VkPushConstantsInfo* i) {
        push_constants(state(d), i->offset, i->size, i->pValues);
        d.CmdPushConstants2(cb, i);
    }
    static void CmdPushConstants2KHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                     const VkPushConstantsInfo* i) {
        push_constants(state(d), i->offset, i->size, i->pValues);
        d.CmdPushConstants2KHR(cb, i);
    }
    // stencil, blend constants and depth bias values set per draw
    static void stencil_set(CbState& s, VkStencilFaceFlags faces, uint32_t VkStencilOpState::*field, uint32_t v) {
        if (faces & VK_STENCIL_FACE_FRONT_BIT) s.dyn.face[0].*field = v;
        if (faces & VK_STENCIL_FACE_BACK_BIT) s.dyn.face[1].*field = v;
    }
    static void CmdSetStencilCompareMask(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkStencilFaceFlags f,
                                         uint32_t v) {
        stencil_set(state(d), f, &VkStencilOpState::compareMask, v);
        d.CmdSetStencilCompareMask(cb, f, v);
    }
    static void CmdSetStencilWriteMask(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkStencilFaceFlags f,
                                       uint32_t v) {
        stencil_set(state(d), f, &VkStencilOpState::writeMask, v);
        d.CmdSetStencilWriteMask(cb, f, v);
    }
    static void CmdSetStencilReference(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkStencilFaceFlags f,
                                       uint32_t v) {
        stencil_set(state(d), f, &VkStencilOpState::reference, v);
        d.CmdSetStencilReference(cb, f, v);
    }
    static void stencil_op(CbState& s, VkStencilFaceFlags faces, VkStencilOp fail, VkStencilOp pass, VkStencilOp depthFail,
                           VkCompareOp compare) {
        for (int i = 0; i < 2; i++)
            if (faces & (i ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT)) {
                VkStencilOpState& o = s.dyn.face[i];
                o.failOp = fail, o.passOp = pass, o.depthFailOp = depthFail, o.compareOp = compare;
            }
    }
    static void CmdSetStencilOp(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkStencilFaceFlags f,
                                VkStencilOp fail, VkStencilOp pass, VkStencilOp depthFail, VkCompareOp compare) {
        stencil_op(state(d), f, fail, pass, depthFail, compare);
        d.CmdSetStencilOp(cb, f, fail, pass, depthFail, compare);
    }
    static void CmdSetStencilOpEXT(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkStencilFaceFlags f,
                                   VkStencilOp fail, VkStencilOp pass, VkStencilOp depthFail, VkCompareOp compare) {
        stencil_op(state(d), f, fail, pass, depthFail, compare);
        d.CmdSetStencilOpEXT(cb, f, fail, pass, depthFail, compare);
    }
    static void CmdSetBlendConstants(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, const float c[4]) {
        memcpy(state(d).dyn.blend, c, sizeof(float) * 4);
        d.CmdSetBlendConstants(cb, c);
    }
    static void CmdSetDepthBias(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, float constant, float clamp,
                                float slope) {
        float* b = state(d).dyn.bias;
        b[0] = constant, b[1] = clamp, b[2] = slope;
        d.CmdSetDepthBias(cb, constant, clamp, slope);
    }
    static void CmdSetViewport(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t first,
                               uint32_t count, const VkViewport* v) {
        if (first == 0 && count) state(d).viewport = v[0];
        d.CmdSetViewport(cb, first, count, v);
    }
    static void CmdSetViewportWithCount(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t count,
                                        const VkViewport* v) {
        if (count) state(d).viewport = v[0];
        d.CmdSetViewportWithCount(cb, count, v);
    }
    static void CmdSetViewportWithCountEXT(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                           uint32_t count, const VkViewport* v) {
        if (count) state(d).viewport = v[0];
        d.CmdSetViewportWithCountEXT(cb, count, v);
    }
    static void CmdSetScissor(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t first,
                              uint32_t count, const VkRect2D* r) {
        if (first == 0 && count) state(d).scissor = r[0];
        d.CmdSetScissor(cb, first, count, r);
    }
    static void CmdSetScissorWithCount(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t count,
                                       const VkRect2D* r) {
        if (count) state(d).scissor = r[0];
        d.CmdSetScissorWithCount(cb, count, r);
    }
    static void CmdSetScissorWithCountEXT(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                          uint32_t count, const VkRect2D* r) {
        if (count) state(d).scissor = r[0];
        state(d).scissor_ext = true;
        d.CmdSetScissorWithCountEXT(cb, count, r);
    }

    static void CmdBeginRendering(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                  const VkRenderingInfo* info) {
        begin_rendering(d, info);
        d.CmdBeginRendering(cb, info);
    }
    static void CmdBeginRenderingKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                     const VkRenderingInfo* info) {
        begin_rendering(d, info);
        d.CmdBeginRenderingKHR(cb, info);
    }
    static void CmdEndRendering(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb) {
        d.CmdEndRendering(cb);
        end_pass(d, cb);
    }
    static void CmdEndRenderingKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb) {
        d.CmdEndRenderingKHR(cb);
        end_pass(d, cb);
    }
    static void CmdNextSubpass(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkSubpassContents c) {
        d.CmdNextSubpass(cb, c);
        next_subpass(d);
    }
    static void CmdNextSubpass2(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, const VkSubpassBeginInfo* b,
                                const VkSubpassEndInfo* e) {
        d.CmdNextSubpass2(cb, b, e);
        next_subpass(d);
    }
    static void CmdNextSubpass2KHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                   const VkSubpassBeginInfo* b, const VkSubpassEndInfo* e) {
        d.CmdNextSubpass2KHR(cb, b, e);
        next_subpass(d);
    }
    static void CmdBeginRenderPass(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                   const VkRenderPassBeginInfo* info, VkSubpassContents contents) {
        begin_render_pass(d, info);
        d.CmdBeginRenderPass(cb, info, contents);
    }
    static void CmdBeginRenderPass2(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                    const VkRenderPassBeginInfo* info, const VkSubpassBeginInfo* sub) {
        begin_render_pass(d, info);
        d.CmdBeginRenderPass2(cb, info, sub);
    }
    static void CmdBeginRenderPass2KHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                       const VkRenderPassBeginInfo* info, const VkSubpassBeginInfo* sub) {
        begin_render_pass(d, info);
        d.CmdBeginRenderPass2KHR(cb, info, sub);
    }
    static void CmdEndRenderPass(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb) {
        d.CmdEndRenderPass(cb);
        end_pass(d, cb);
    }
    static void CmdEndRenderPass2(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                  const VkSubpassEndInfo* info) {
        d.CmdEndRenderPass2(cb, info);
        end_pass(d, cb);
    }
    static void CmdEndRenderPass2KHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                     const VkSubpassEndInfo* info) {
        d.CmdEndRenderPass2KHR(cb, info);
        end_pass(d, cb);
    }

    static void CmdBeginDebugUtilsLabelEXT(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                           const VkDebugUtilsLabelEXT* label) {
        begin_label(cb, label);
        d.CmdBeginDebugUtilsLabelEXT(cb, label);
    }
    // dynamic state (core 1.3 names and their EXT aliases, which vkroots hooks separately)
#define FIXFRAME_DYN(Name, Type, field)                                                                         \
    static void Cmd##Name(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, Type v) {              \
        state(d).dyn.field = v;                                                                                 \
        d.Cmd##Name(cb, v);                                                                                     \
    }                                                                                                           \
    static void Cmd##Name##EXT(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, Type v) {         \
        state(d).dyn.field = v;                                                                                 \
        d.Cmd##Name##EXT(cb, v);                                                                                \
    }
    FIXFRAME_DYN(SetCullMode, VkCullModeFlags, cull)
    FIXFRAME_DYN(SetFrontFace, VkFrontFace, front)
    FIXFRAME_DYN(SetPrimitiveTopology, VkPrimitiveTopology, topology)
    FIXFRAME_DYN(SetDepthTestEnable, VkBool32, depth_test)
    FIXFRAME_DYN(SetDepthWriteEnable, VkBool32, depth_write)
    FIXFRAME_DYN(SetDepthCompareOp, VkCompareOp, compare)
    FIXFRAME_DYN(SetDepthBiasEnable, VkBool32, depth_bias)
    FIXFRAME_DYN(SetStencilTestEnable, VkBool32, stencil)
#undef FIXFRAME_DYN

    static void CmdEndDebugUtilsLabelEXT(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb) {
        end_label(cb);
        d.CmdEndDebugUtilsLabelEXT(cb);
    }

    static void CmdDraw(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t vertices,
                        uint32_t instances, uint32_t firstVertex, uint32_t firstInstance) {
        if (begin_draw(d, cb, kDraw, vertices, instances)) d.CmdDraw(cb, vertices, instances, firstVertex, firstInstance);
        end_draw(d, cb);
    }
    static void CmdDrawIndexed(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t indices,
                               uint32_t instances, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
        if (begin_draw(d, cb, kIndexed, indices, instances)) d.CmdDrawIndexed(cb, indices, instances, firstIndex, vertexOffset, firstInstance);
        end_draw(d, cb);
    }
    static void CmdDrawIndirect(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                VkDeviceSize offset, uint32_t count, uint32_t stride) {
        if (begin_draw(d, cb, kIndirect, count, 0)) d.CmdDrawIndirect(cb, buffer, offset, count, stride);
        end_draw(d, cb);
    }
    static void CmdDrawIndexedIndirect(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                       VkDeviceSize offset, uint32_t count, uint32_t stride) {
        if (begin_draw(d, cb, kIndexedIndirect, count, 0)) d.CmdDrawIndexedIndirect(cb, buffer, offset, count, stride);
        end_draw(d, cb);
    }
    static void CmdDrawIndirectCount(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                     VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countOffset,
                                     uint32_t maxCount, uint32_t stride) {
        if (begin_draw(d, cb, kIndirectCount, maxCount, 0)) d.CmdDrawIndirectCount(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
        end_draw(d, cb);
    }
    static void CmdDrawIndirectCountKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                        VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countOffset,
                                        uint32_t maxCount, uint32_t stride) {
        if (begin_draw(d, cb, kIndirectCount, maxCount, 0)) d.CmdDrawIndirectCountKHR(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
        end_draw(d, cb);
    }
    static void CmdDrawIndexedIndirectCount(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                            VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer,
                                            VkDeviceSize countOffset, uint32_t maxCount, uint32_t stride) {
        if (begin_draw(d, cb, kIndexedIndirectCount, maxCount, 0)) d.CmdDrawIndexedIndirectCount(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
        end_draw(d, cb);
    }
    static void CmdDrawIndexedIndirectCountKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                               VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer,
                                               VkDeviceSize countOffset, uint32_t maxCount, uint32_t stride) {
        if (begin_draw(d, cb, kIndexedIndirectCount, maxCount, 0)) d.CmdDrawIndexedIndirectCountKHR(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
        end_draw(d, cb);
    }
    static void CmdDispatch(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t x, uint32_t y,
                            uint32_t z) {
        if (begin_draw(d, cb, kDispatch, x, y)) d.CmdDispatch(cb, x, y, z);
        end_draw(d, cb);
    }
    static void CmdDispatchIndirect(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                    VkDeviceSize offset) {
        if (begin_draw(d, cb, kDispatchIndirect, 0, 0)) d.CmdDispatchIndirect(cb, buffer, offset);
        end_draw(d, cb);
    }
    static void CmdExecuteCommands(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t count,
                                   const VkCommandBuffer* secondaries) {
        CbState& s = state(d);
        for (uint32_t i = 0; i < count; i++) {
            Rec r{kExecute};
            r.pass = s.pass;
            r.secondary = secondaries[i];
            s.recs.push_back(r);
        }
        d.CmdExecuteCommands(cb, count, secondaries);
    }

    // --- submits and presents ---
    static VkResult QueueSubmit(const vkroots::VkQueueDispatch& d, VkQueue queue, uint32_t count,
                                const VkSubmitInfo* submits, VkFence fence) {
        for (uint32_t i = 0; i < count; i++) submitted(submits[i].pCommandBuffers, submits[i].commandBufferCount);
        return d.QueueSubmit(queue, count, submits, fence);
    }
    static void submitted2(uint32_t count, const VkSubmitInfo2* submits) {
        for (uint32_t i = 0; i < count; i++) {
            std::vector<VkCommandBuffer> cbs;
            for (uint32_t j = 0; j < submits[i].commandBufferInfoCount; j++)
                cbs.push_back(submits[i].pCommandBufferInfos[j].commandBuffer);
            submitted(cbs.data(), (uint32_t)cbs.size());
        }
    }
    static VkResult QueueSubmit2(const vkroots::VkQueueDispatch& d, VkQueue queue, uint32_t count,
                                 const VkSubmitInfo2* submits, VkFence fence) {
        submitted2(count, submits);
        return d.QueueSubmit2(queue, count, submits, fence);
    }
    static VkResult QueueSubmit2KHR(const vkroots::VkQueueDispatch& d, VkQueue queue, uint32_t count,
                                    const VkSubmitInfo2* submits, VkFence fence) {
        submitted2(count, submits);
        return d.QueueSubmit2KHR(queue, count, submits, fence);
    }

    // Frame boundaries: a capture or job armed here covers the frame recorded and submitted until the next
    // present, where its results are read back.
    static VkResult QueuePresentKHR(const vkroots::VkQueueDispatch& d, VkQueue queue, const VkPresentInfoKHR* info) {
        uint64_t frame = ++g_frame;
        core::frame_done(frame, g_frame_draws.exchange(0));
        std::unique_ptr<Capture> done;
        std::unique_ptr<Job> job;
        std::vector<std::pair<VkQueryPool, uint32_t>> slots;
        {
            std::lock_guard lock(g_capture_mutex);
            auto arm_capture = [&](const Request& req) {
                g_building = std::make_unique<Capture>();
                g_building->frame = frame + 1;
                g_building->picked = req.kind == Request::kPick;
                g_building->pick_x = req.x, g_building->pick_y = req.y;
            };
            Request req;
            if (g_running && ++g_running->presents == 1) {
                if (g_running->req.kind == Request::kPick) arm_capture(g_running->req);
            } else if (g_running) {  // the job's frame is submitted: read it back
                done = std::move(g_building);
                slots = std::move(g_pick_slots);
                job = std::move(g_running);
                g_job = nullptr;
            } else if (g_building) {  // a plain capture: the frame since the last present
                done = std::move(g_building);
            } else if (core::take_request(req)) {
                if (req.kind == Request::kCapture) {
                    arm_capture(req);
                } else {
                    g_running = std::make_unique<Job>();
                    g_running->req = req;
                    g_job = g_running.get();
                }
            }
        }
        const vkroots::VkDeviceDispatch& dd = *d.pDeviceDispatch;
        retire(dd, frame);
        if (!done && !job) return d.QueuePresentKHR(queue, info);

        // a pick frame's image is the pixel alone: not worth reading back
        VkResult r = done && !done->picked ? present_with_readback(d, queue, info, done->png) : d.QueuePresentKHR(queue, info);
        if (job) d.QueueWaitIdle(queue);  // the job's frame is done (on this queue)
        if (done && done->picked)
            for (size_t i = 0; i < slots.size() && i < done->draws.size(); i++) {
                if (!slots[i].first) continue;
                uint64_t v[2] = {};  // samples, availability
                if (dd.GetQueryPoolResults(dd.Device, slots[i].first, slots[i].second, 1, sizeof v, v, sizeof v,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS &&
                    v[1])
                    done->draws[i].samples = (int64_t)v[0];
            }
        if (job && job->req.kind == Request::kScrub) {
            std::string png, error = job->error;
            if (!job->found) error = "That draw was not recorded in the next frame (the frame changed): capture again.";
            else if (error.empty() && !job->target) error = "Its pass did not end in the same command buffer.";
            else if (error.empty())
                if (const uint8_t* p = job->target->map()) png = picture(job->format, p, job->extent.width, job->extent.height);
            if (png.empty() && error.empty()) error = "The copy could not be read.";
            core::scrubbed(std::move(png), std::move(error));
        }
        if (done) {
            fprintf(stderr, "[fixframe] %s frame %llu: %zu draws, %zu passes, %zu textures%s\n",
                    done->picked ? "picked" : "captured", (unsigned long long)done->frame, done->draws.size(),
                    done->passes.size(), done->textures.size(), done->png.empty() ? ", no image" : "");
            core::publish(std::move(done));
        }
        if (job) {
            job->retire_at = frame + 3;
            g_retired.push_back(std::move(job));
        }
        return r;
    }
};

}  // namespace
}  // namespace fixframe

namespace fixframe {
namespace {
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance_proc_addr(VkInstance instance, const char* name) {
    if (!strcmp(name, "vkSetDebugUtilsObjectNameEXT") && g_next_set_name) return (PFN_vkVoidFunction)&hook_set_name;
    if (!strcmp(name, "vkCmdBeginDebugUtilsLabelEXT") && g_next_begin_label) return (PFN_vkVoidFunction)&hook_begin_label;
    if (!strcmp(name, "vkCmdEndDebugUtilsLabelEXT") && g_next_end_label) return (PFN_vkVoidFunction)&hook_end_label;
    return vkroots::GetInstanceProcAddr<Instance, Device>(instance, name);
}
}  // namespace
}  // namespace fixframe

VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkNegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* v) {
    VkResult r = vkroots::NegotiateLoaderLayerInterfaceVersion<fixframe::Instance, fixframe::Device>(v);
    v->pfnGetInstanceProcAddr = &fixframe::get_instance_proc_addr;
    return r;
}
