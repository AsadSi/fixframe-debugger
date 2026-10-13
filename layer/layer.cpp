// Copyright (c) 2026 asadsi. MIT licence (see LICENSE).
//
// VK_LAYER_FIXFRAME_debug. M0: proves the layer loads (logs instance and device creation) and counts the
// draws recorded between presents. Draws are counted as recorded, so a command buffer recorded once and
// submitted many times counts once; M1 moves the count to submits.
#include "vkroots.h"

#include <atomic>
#include <cstdio>

namespace fixframe {

static std::atomic<uint64_t> g_draws{0};
static uint64_t g_frame = 0;  // presents seen (one presenting queue is assumed for M0)

static const char* arch() {
#if defined(__x86_64__)
    return "x86_64";
#elif defined(__aarch64__)
    return "arm64";
#else
    return "?";
#endif
}

struct Instance {
    static VkResult CreateInstance(PFN_vkCreateInstance next, const VkInstanceCreateInfo* info,
                                   const VkAllocationCallbacks* alloc, VkInstance* instance) {
        const char* app = info->pApplicationInfo && info->pApplicationInfo->pApplicationName
                              ? info->pApplicationInfo->pApplicationName : "?";
        fprintf(stderr, "[fixframe] loaded (%s) into \"%s\"\n", arch(), app);
        return next(info, alloc, instance);
    }

    static VkResult CreateDevice(const vkroots::VkPhysicalDeviceDispatch& d, VkPhysicalDevice gpu,
                                 const VkDeviceCreateInfo* info, const VkAllocationCallbacks* alloc, VkDevice* device) {
        VkPhysicalDeviceProperties p;
        d.GetPhysicalDeviceProperties(gpu, &p);
        fprintf(stderr, "[fixframe] device: %s, Vulkan %u.%u.%u\n", p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
                VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        return d.CreateDevice(gpu, info, alloc, device);
    }
};

struct Device {
    static void CmdDraw(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t vertices,
                        uint32_t instances, uint32_t firstVertex, uint32_t firstInstance) {
        g_draws.fetch_add(1, std::memory_order_relaxed);
        d.CmdDraw(cb, vertices, instances, firstVertex, firstInstance);
    }
    static void CmdDrawIndexed(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, uint32_t indices,
                               uint32_t instances, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance) {
        g_draws.fetch_add(1, std::memory_order_relaxed);
        d.CmdDrawIndexed(cb, indices, instances, firstIndex, vertexOffset, firstInstance);
    }
    static void CmdDrawIndirect(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                VkDeviceSize offset, uint32_t count, uint32_t stride) {
        g_draws.fetch_add(1, std::memory_order_relaxed);
        d.CmdDrawIndirect(cb, buffer, offset, count, stride);
    }
    static void CmdDrawIndexedIndirect(const vkroots::VkCommandBufferDispatch& d, VkCommandBuffer cb, VkBuffer buffer,
                                       VkDeviceSize offset, uint32_t count, uint32_t stride) {
        g_draws.fetch_add(1, std::memory_order_relaxed);
        d.CmdDrawIndexedIndirect(cb, buffer, offset, count, stride);
    }

    static VkResult QueuePresentKHR(const vkroots::VkQueueDispatch& d, VkQueue queue, const VkPresentInfoKHR* info) {
        uint64_t draws = g_draws.exchange(0, std::memory_order_relaxed);
        if (++g_frame % 120 == 0) fprintf(stderr, "[fixframe] frame %llu: %llu draws\n", (unsigned long long)g_frame,
                                          (unsigned long long)draws);
        return d.QueuePresentKHR(queue, info);
    }
};

}  // namespace fixframe

VKROOTS_DEFINE_LAYER_INTERFACES(fixframe::Instance, fixframe::Device)
