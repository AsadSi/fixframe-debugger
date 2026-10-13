// Copyright (c) 2026 asadsi. MIT licence (see LICENSE).
#include "fixframe_sdk.h"

#include <memory>

namespace fixframe::sdk {
namespace {
std::unique_ptr<Capture> g_building;  // render thread only
}

void start(const std::string& app, const std::string& device, const std::string& backend) {
    core::start(app, device, backend);
}

bool capturing() { return g_building != nullptr; }
Capture* capture() { return g_building.get(); }

int pass(const Pass& p) {
    auto& passes = g_building->passes;
    if (passes.empty() || passes.back().colors != p.colors || passes.back().depth != p.depth ||
        passes.back().width != p.width || passes.back().height != p.height)
        passes.push_back(p);
    return (int)passes.size() - 1;
}

void frame_end(uint64_t frame, uint32_t draws, const std::function<std::string()>& image) {
    core::frame_done(frame, draws);
    if (g_building) {
        g_building->png = image();
        fprintf(stderr, "[fixframe] captured frame %llu: %zu draws, %zu passes, %zu textures\n",
                (unsigned long long)g_building->frame, g_building->draws.size(), g_building->passes.size(),
                g_building->textures.size());
        core::publish(std::move(g_building));
        return;
    }
    Request r;
    if (!core::take_request(r)) return;
    if (r.kind == Request::kCapture) {
        g_building = std::make_unique<Capture>();
        g_building->frame = frame + 1;
    } else {
        core::failed("This renderer reports draws through Fixframe's SDK, which can't pick pixels or scrub yet: "
                     "capture, compare and save work.");
    }
}

}  // namespace fixframe::sdk
