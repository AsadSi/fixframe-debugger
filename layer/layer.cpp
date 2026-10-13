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
#include <shared_mutex>
#include <string_view>
#include <unordered_map>

#include "../core/core.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"

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
};
struct SwapchainInfo {
    VkFormat format;
    VkExtent2D extent;
    bool readable;  // made with TRANSFER_SRC (we add it when the surface allows)
    std::vector<VkImage> images;
};

// Dynamic state the layer follows, as bits; a pipeline's mask says which of them it takes from the command
// buffer instead of its own create info. The keys match pipeline_state()'s, so a draw's values replace them.
enum DynBit : uint16_t { kCull = 1, kFront = 2, kTopology = 4, kDepthTest = 8, kDepthWrite = 16, kDepthCompare = 32,
                         kDepthBias = 64, kStencil = 128 };
struct Dyn {
    VkCullModeFlags cull = 0;
    VkFrontFace front = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCompareOp compare = VK_COMPARE_OP_NEVER;
    VkBool32 depth_test = 0, depth_write = 0, depth_bias = 0, stencil = 0;
};

struct PipelineRec {
    std::shared_ptr<const Program> program;  // replaced, not changed, when named: captures hold the old one
    uint16_t dynamic = 0;
};

std::shared_mutex g_objects;
std::unordered_map<uint64_t, std::string> g_names;  // any object, from vkSetDebugUtilsObjectNameEXT
std::unordered_map<VkShaderModule, Shader> g_modules;
std::unordered_map<VkPipeline, PipelineRec> g_pipelines;
std::unordered_map<VkImage, ImageInfo> g_images;
std::unordered_map<VkImageView, VkImage> g_views;
std::unordered_map<VkFramebuffer, std::vector<VkImageView>> g_framebuffers;
std::unordered_map<VkDescriptorSet, std::map<uint64_t, VkImageView>> g_sets;  // (binding << 32 | element) -> view
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

bool is_image_descriptor(VkDescriptorType t) {
    return t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
           t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
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

uint16_t dynamic_mask(const VkPipelineDynamicStateCreateInfo* dy) {
    uint16_t m = 0;
    for (uint32_t i = 0; dy && i < dy->dynamicStateCount; i++) switch (dy->pDynamicStates[i]) {
        case VK_DYNAMIC_STATE_CULL_MODE: m |= kCull; break;
        case VK_DYNAMIC_STATE_FRONT_FACE: m |= kFront; break;
        case VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY: m |= kTopology; break;
        case VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE: m |= kDepthTest; break;
        case VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE: m |= kDepthWrite; break;
        case VK_DYNAMIC_STATE_DEPTH_COMPARE_OP: m |= kDepthCompare; break;
        case VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE: m |= kDepthBias; break;
        case VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE: m |= kStencil; break;
        default: break;
    }
    return m;
}

void dynamic_state(std::vector<KV>& out, uint16_t mask, const Dyn& d) {
    if (mask & kTopology) out.push_back({"topology", strip(string_VkPrimitiveTopology(d.topology), "VK_PRIMITIVE_TOPOLOGY_")});
    if (mask & kCull) out.push_back({"cull", cull_text(d.cull)});
    if (mask & kFront) out.push_back({"front face", strip(string_VkFrontFace(d.front), "VK_FRONT_FACE_")});
    if (mask & kDepthBias) out.push_back({"depth bias", on(d.depth_bias)});
    if (mask & kDepthTest) out.push_back({"depth test", on(d.depth_test)});
    if (mask & kDepthWrite) out.push_back({"depth write", on(d.depth_write)});
    if (mask & kDepthCompare) out.push_back({"depth compare", strip(string_VkCompareOp(d.compare), "VK_COMPARE_OP_")});
    if (mask & kStencil) out.push_back({"stencil test", on(d.stencil)});
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
    }
    if (auto* cb = ci.pColorBlendState) {
        for (uint32_t i = 0; i < cb->attachmentCount; i++) {
            s.push_back({"blend " + std::to_string(i), blend_text(cb->pAttachments[i])});
            s.push_back({"write mask " + std::to_string(i), mask_text(cb->pAttachments[i].colorWriteMask)});
        }
        if (cb->logicOpEnable) s.push_back({"logic op", strip(string_VkLogicOp(cb->logicOp), "VK_LOGIC_OP_")});
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
    std::vector<std::pair<uint64_t, VkImageView>> pushed;  // (set << 48 | binding << 32 | element) -> view
};

struct PassRec {
    std::vector<VkImageView> views;  // colours then depth (told apart by format at capture)
    VkExtent2D extent;
    uint32_t label;
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
    Dyn dyn;

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

void record(const vkroots::VkCommandBufferDispatch& d, Kind kind, uint32_t a, uint32_t b) {
    CbState& s = state(d);
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
    s.recs.push_back(r);
    s.draws++;
}

void push_descriptors(const vkroots::VkCommandBufferDispatch& d, VkPipelineBindPoint bp, uint32_t set, uint32_t count,
                      const VkWriteDescriptorSet* writes) {
    CbState& s = state(d);
    Bindings& b = s.cur[point(bp)];
    std::erase_if(b.pushed, [&](auto& p) { return (p.first >> 48) == set; });
    b.sets[set & 7] = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < count; i++) {
        const VkWriteDescriptorSet& w = writes[i];
        if (!is_image_descriptor(w.descriptorType) || !w.pImageInfo) continue;
        for (uint32_t e = 0; e < w.descriptorCount; e++)
            b.pushed.push_back({(uint64_t)set << 48 | (uint64_t)w.dstBinding << 32 | (w.dstArrayElement + e),
                                w.pImageInfo[e].imageView});
    }
    s.bound[point(bp)] = -1;
}

void begin_pass(const vkroots::VkCommandBufferDispatch& d, std::vector<VkImageView> views, VkExtent2D extent) {
    CbState& s = state(d);
    s.passes.push_back({std::move(views), extent, s.labels.empty() ? 0 : s.labels.back()});
    s.pass = (int)s.passes.size() - 1;
}

void begin_rendering(const vkroots::VkCommandBufferDispatch& d, const VkRenderingInfo* info) {
    std::vector<VkImageView> views;
    for (uint32_t i = 0; i < info->colorAttachmentCount; i++)
        if (info->pColorAttachments[i].imageView) views.push_back(info->pColorAttachments[i].imageView);
    if (info->pDepthAttachment && info->pDepthAttachment->imageView) views.push_back(info->pDepthAttachment->imageView);
    else if (info->pStencilAttachment && info->pStencilAttachment->imageView)
        views.push_back(info->pStencilAttachment->imageView);
    begin_pass(d, std::move(views), info->renderArea.extent);
}

void begin_render_pass(const vkroots::VkCommandBufferDispatch& d, const VkRenderPassBeginInfo* info) {
    std::vector<VkImageView> views;
    if (auto* imageless = find_in_chain<VkRenderPassAttachmentBeginInfo>(info->pNext,
                                                                         VK_STRUCTURE_TYPE_RENDER_PASS_ATTACHMENT_BEGIN_INFO)) {
        views.assign(imageless->pAttachments, imageless->pAttachments + imageless->attachmentCount);
    } else {
        std::shared_lock lock(g_objects);
        auto it = g_framebuffers.find(info->framebuffer);
        if (it != g_framebuffers.end()) views = it->second;
    }
    begin_pass(d, std::move(views), info->renderArea.extent);
}

// ---------------------------------------------------------------------------------------------------------
// Frames and captures

std::atomic<uint64_t> g_frame{0};
std::atomic<uint32_t> g_frame_draws{0};
std::mutex g_capture_mutex;              // the capture being built (submits can come from several threads)
std::unique_ptr<Capture> g_building;     // non-null while the frame being captured is submitted

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

uint64_t add_texture(Capture& c, VkImageView view) {  // g_objects held
    auto v = g_views.find(view);
    if (v == g_views.end()) return 0;
    VkImage image = v->second;
    uint64_t key = (uint64_t)image;
    if (!c.textures.count(key)) {
        Texture t;
        t.id = key;
        t.name = name_of(key);
        if (auto it = g_images.find(image); it != g_images.end()) {
            t.format = strip(string_VkFormat(it->second.format), "VK_FORMAT_");
            t.width = it->second.extent.width, t.height = it->second.extent.height;
            t.mips = it->second.mips, t.layers = it->second.layers;
        }
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
// secondary's draws. g_objects held (shared).
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
        out.count = r.a, out.instances = r.b;
        out.pass = r.kind >= kDispatch ? -1 : pass;
        out.label = label_text(r.label);
        if (auto it = g_pipelines.find(r.pipeline); it != g_pipelines.end()) {
            out.program = it->second.program;
            dynamic_state(out.dynamic, it->second.dynamic, r.dyn);
        }
        if (r.kind < kDispatch) {
            char b[96];
            snprintf(b, sizeof b, "%g %g %g x %g", r.viewport.x, r.viewport.y, r.viewport.width, r.viewport.height);
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
            for (uint32_t set = 0; set < 8; set++) {
                auto it = b.sets[set] ? g_sets.find(b.sets[set]) : g_sets.end();
                if (it == g_sets.end()) continue;
                for (auto& [k, view] : it->second)
                    if (uint64_t t = add_texture(c, view))
                        out.textures.push_back({slot(set, k >> 32, k & 0xffffffff), std::to_string(t)});
            }
            for (auto& [k, view] : b.pushed)
                if (uint64_t t = add_texture(c, view))
                    out.textures.push_back({slot(k >> 48, (k >> 32) & 0xffff, k & 0xffffffff), std::to_string(t)});
        }
        c.draws.push_back(std::move(out));
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

// ---------------------------------------------------------------------------------------------------------
// Reading the presented image back (only for a captured frame). The copy is submitted on the present queue,
// waiting for what the present waited for; the present then waits for the copy.

struct DeviceInfo {
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties memory{};
    std::unordered_map<VkQueue, uint32_t> queue_family;
};
DeviceInfo g_device;

void write_png(void* ctx, void* data, int size) { static_cast<std::string*>(ctx)->append((const char*)data, size); }

std::string encode(const uint8_t* src, VkFormat format, uint32_t w, uint32_t h) {
    std::vector<uint8_t> rgba(size_t(w) * h * 4);
    for (size_t i = 0; i < size_t(w) * h; i++) {
        const uint8_t* p = src + i * 4;
        uint8_t* o = &rgba[i * 4];
        switch (format) {
            case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
                o[0] = p[2], o[1] = p[1], o[2] = p[0];
                break;
            case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
                o[0] = p[0], o[1] = p[1], o[2] = p[2];
                break;
            case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: {
                uint32_t v;
                memcpy(&v, p, 4);
                uint8_t r = (v & 1023) >> 2, g = ((v >> 10) & 1023) >> 2, b = ((v >> 20) & 1023) >> 2;
                bool abgr = format == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
                o[0] = abgr ? r : b, o[1] = g, o[2] = abgr ? b : r;
                break;
            }
            default:
                return {};
        }
        o[3] = 255;
    }
    std::string png;
    stbi_write_png_to_func(write_png, &png, (int)w, (int)h, 4, rgba.data(), (int)w * 4);
    return png;
}

bool readable_format(VkFormat f) {
    return f == VK_FORMAT_B8G8R8A8_UNORM || f == VK_FORMAT_B8G8R8A8_SRGB || f == VK_FORMAT_R8G8B8A8_UNORM ||
           f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 ||
           f == VK_FORMAT_A2R10G10B10_UNORM_PACK32;
}

// Presents with the swapchain image copied to the host first; png gets the image (empty on failure).
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
    if (!sc.readable || !readable_format(sc.format) || index >= sc.images.size() || family == g_device.queue_family.end())
        return qd.QueuePresentKHR(queue, info);
    VkImage image = sc.images[index];
    VkDeviceSize size = VkDeviceSize(sc.extent.width) * sc.extent.height * 4;

    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkSemaphore done = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    auto cleanup = [&] {
        if (fence) d.DestroyFence(dev, fence, nullptr);
        if (done) d.DestroySemaphore(dev, done, nullptr);
        if (pool) d.DestroyCommandPool(dev, pool, nullptr);
        if (buffer) d.DestroyBuffer(dev, buffer, nullptr);
        if (memory) d.FreeMemory(dev, memory, nullptr);
    };

    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryRequirements req;
    uint32_t type = UINT32_MAX;
    if (d.CreateBuffer(dev, &bci, nullptr, &buffer) == VK_SUCCESS) {
        d.GetBufferMemoryRequirements(dev, buffer, &req);
        const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        for (uint32_t i = 0; i < g_device.memory.memoryTypeCount && type == UINT32_MAX; i++)
            if ((req.memoryTypeBits & (1u << i)) && (g_device.memory.memoryTypes[i].propertyFlags & want) == want) type = i;
    }
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, nullptr, req.size, type};
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, family->second};
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (type == UINT32_MAX || d.AllocateMemory(dev, &mai, nullptr, &memory) != VK_SUCCESS ||
        d.BindBufferMemory(dev, buffer, memory, 0) != VK_SUCCESS ||
        d.CreateCommandPool(dev, &pci, nullptr, &pool) != VK_SUCCESS ||
        d.CreateSemaphore(dev, &sci, nullptr, &done) != VK_SUCCESS || d.CreateFence(dev, &fci, nullptr, &fence) != VK_SUCCESS) {
        cleanup();
        return qd.QueuePresentKHR(queue, info);
    }
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, pool,
                                    VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    d.AllocateCommandBuffers(dev, &cai, &cb);
    *reinterpret_cast<void**>(cb) = *reinterpret_cast<void**>(dev);  // a layer's own dispatchable object: the loader's table

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                   VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    d.BeginCommandBuffer(cb, &begin);
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {sc.extent.width, sc.extent.height, 1};
    d.CmdCopyImageToBuffer(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &copy);
    std::swap(barrier.oldLayout, barrier.newLayout);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = 0;
    d.CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
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
        void* mapped = nullptr;
        if (d.WaitForFences(dev, 1, &fence, VK_TRUE, 5'000'000'000ull) == VK_SUCCESS &&
            d.MapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, &mapped) == VK_SUCCESS) {
            png = encode(static_cast<const uint8_t*>(mapped), sc.format, sc.extent.width, sc.extent.height);
            d.UnmapMemory(dev, memory);
        }
    }
    cleanup();
    return result;
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
            g_images[*image] = {info->format, info->extent, info->mipLevels, info->arrayLayers};
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
        if (r == VK_SUCCESS && info->pAttachments) {
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
                    set[(uint64_t)w.dstBinding << 32 | (w.dstArrayElement + e)] = w.pImageInfo[e].imageView;
            }
            for (uint32_t i = 0; i < copyCount; i++) {
                const VkCopyDescriptorSet& c = copies[i];
                auto src = g_sets.find(c.srcSet);
                if (src == g_sets.end()) continue;
                for (uint32_t e = 0; e < c.descriptorCount; e++) {
                    auto it = src->second.find((uint64_t)c.srcBinding << 32 | (c.srcArrayElement + e));
                    if (it != src->second.end())
                        g_sets[c.dstSet][(uint64_t)c.dstBinding << 32 | (c.dstArrayElement + e)] = it->second;
                }
            }
        }
        d.UpdateDescriptorSets(dev, writeCount, writes, copyCount, copies);
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
                g_images[images[i]] = {sc.format, {sc.extent.width, sc.extent.height, 1}, 1, 1};
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
        return d.BeginCommandBuffer(cb, info);
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
        CbState& s = state(d);
        for (uint32_t i = 0; i < count && first + i < 8; i++) {
            s.cur[point(bp)].sets[first + i] = sets[i];
            std::erase_if(s.cur[point(bp)].pushed, [&](auto& p) { return (p.first >> 48) == first + i; });
        }
        s.bound[point(bp)] = -1;
        d.CmdBindDescriptorSets(cb, bp, layout, first, count, sets, dynCount, dyn);
    }
    static void CmdPushDescriptorSetKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                        VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t count,
                                        const VkWriteDescriptorSet* writes) {
        push_descriptors(d, bp, set, count, writes);
        d.CmdPushDescriptorSetKHR(cb, bp, layout, set, count, writes);
    }
    static void CmdPushDescriptorSet(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                     VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t count,
                                     const VkWriteDescriptorSet* writes) {
        push_descriptors(d, bp, set, count, writes);
        d.CmdPushDescriptorSet(cb, bp, layout, set, count, writes);
    }
    static void CmdPushConstants(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkPipelineLayout layout,
                                 VkShaderStageFlags stages, uint32_t offset, uint32_t size, const void* values) {
        CbState& s = state(d);
        if (s.push_data.size() < offset + size) s.push_data.resize(offset + size);
        memcpy(s.push_data.data() + offset, values, size);
        s.push = -1;
        d.CmdPushConstants(cb, layout, stages, offset, size, values);
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
        state(d).pass = -1;
        d.CmdEndRendering(cb);
    }
    static void CmdEndRenderingKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb) {
        state(d).pass = -1;
        d.CmdEndRenderingKHR(cb);
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
        state(d).pass = -1;
        d.CmdEndRenderPass(cb);
    }
    static void CmdEndRenderPass2(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                  const VkSubpassEndInfo* info) {
        state(d).pass = -1;
        d.CmdEndRenderPass2(cb, info);
    }
    static void CmdEndRenderPass2KHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                     const VkSubpassEndInfo* info) {
        state(d).pass = -1;
        d.CmdEndRenderPass2KHR(cb, info);
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
        record(d, kDraw, vertices, instances);
        d.CmdDraw(cb, vertices, instances, firstVertex, firstInstance);
    }
    static void CmdDrawIndexed(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t indices,
                               uint32_t instances, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
        record(d, kIndexed, indices, instances);
        d.CmdDrawIndexed(cb, indices, instances, firstIndex, vertexOffset, firstInstance);
    }
    static void CmdDrawIndirect(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                VkDeviceSize offset, uint32_t count, uint32_t stride) {
        record(d, kIndirect, count, 0);
        d.CmdDrawIndirect(cb, buffer, offset, count, stride);
    }
    static void CmdDrawIndexedIndirect(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                       VkDeviceSize offset, uint32_t count, uint32_t stride) {
        record(d, kIndexedIndirect, count, 0);
        d.CmdDrawIndexedIndirect(cb, buffer, offset, count, stride);
    }
    static void CmdDrawIndirectCount(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                     VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countOffset,
                                     uint32_t maxCount, uint32_t stride) {
        record(d, kIndirectCount, maxCount, 0);
        d.CmdDrawIndirectCount(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
    }
    static void CmdDrawIndirectCountKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                        VkDeviceSize offset, VkBuffer countBuffer, VkDeviceSize countOffset,
                                        uint32_t maxCount, uint32_t stride) {
        record(d, kIndirectCount, maxCount, 0);
        d.CmdDrawIndirectCountKHR(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
    }
    static void CmdDrawIndexedIndirectCount(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                            VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer,
                                            VkDeviceSize countOffset, uint32_t maxCount, uint32_t stride) {
        record(d, kIndexedIndirectCount, maxCount, 0);
        d.CmdDrawIndexedIndirectCount(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
    }
    static void CmdDrawIndexedIndirectCountKHR(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb,
                                               VkBuffer buffer, VkDeviceSize offset, VkBuffer countBuffer,
                                               VkDeviceSize countOffset, uint32_t maxCount, uint32_t stride) {
        record(d, kIndexedIndirectCount, maxCount, 0);
        d.CmdDrawIndexedIndirectCountKHR(cb, buffer, offset, countBuffer, countOffset, maxCount, stride);
    }
    static void CmdDispatch(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t x, uint32_t y,
                            uint32_t z) {
        record(d, kDispatch, x, y);
        d.CmdDispatch(cb, x, y, z);
    }
    static void CmdDispatchIndirect(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                    VkDeviceSize offset) {
        record(d, kDispatchIndirect, 0, 0);
        d.CmdDispatchIndirect(cb, buffer, offset);
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

    static VkResult QueuePresentKHR(const vkroots::VkQueueDispatch& d, VkQueue queue, const VkPresentInfoKHR* info) {
        uint64_t frame = ++g_frame;
        core::frame_done(frame, g_frame_draws.exchange(0));
        std::unique_ptr<Capture> done;
        {
            std::lock_guard lock(g_capture_mutex);
            done = std::move(g_building);  // the frame captured ends with this present
            if (!done && core::take_capture_request()) {
                g_building = std::make_unique<Capture>();
                g_building->frame = frame + 1;
            }
        }
        if (!done) return d.QueuePresentKHR(queue, info);
        VkResult r = present_with_readback(d, queue, info, done->png);
        fprintf(stderr, "[fixframe] captured frame %llu: %zu draws, %zu passes, %zu textures%s\n",
                (unsigned long long)done->frame, done->draws.size(), done->passes.size(), done->textures.size(),
                done->png.empty() ? ", no image" : "");
        core::publish(std::move(done));
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
