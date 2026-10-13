// Copyright (c) 2026 asadsi. MIT licence (see LICENSE).
#include "core.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>

#include "httplib.h"
#include "ui_assets.h"  // generated from ui/ (cmake/embed.cmake)

namespace fixframe::core {
namespace {

std::mutex g_mutex;
std::condition_variable g_published;
std::string g_app, g_device, g_backend;
uint64_t g_frame = 0;
uint32_t g_draws = 0;
std::deque<double> g_times;  // present times of the last second, for the frame rate
// One request at a time: the server thread waits for its result (g_done counts results).
bool g_pending = false, g_taken = false;  // asked for; handed to the backend
Request g_request;
uint64_t g_done = 0;
uint32_t g_result_capture = 0;  // kCapture, kPick
std::string g_result_png, g_result_error;  // kScrub
uint32_t g_next_id = 1;
std::deque<std::shared_ptr<const Capture>> g_captures;  // newest last
constexpr size_t kKeep = 8;

double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// --- JSON (writing only; the API takes no JSON in) ---
void str(std::string& o, const std::string& s) {
    o += '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o += '\\', o += c;
        else if (c == '\n') o += "\\n";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += c;
    }
    o += '"';
}
void kvs(std::string& o, const std::vector<KV>& v) {
    o += '[';
    for (size_t i = 0; i < v.size(); i++) {
        if (i) o += ',';
        o += '[';
        str(o, v[i].key);
        o += ',';
        str(o, v[i].value);
        o += ']';
    }
    o += ']';
}
// 64-bit ids go out as strings: JavaScript numbers lose bits past 2^53
std::string id(uint64_t v) { return '"' + std::to_string(v) + '"'; }

std::string capture_json(const Capture& c) {
    std::string o = "{\"id\":" + std::to_string(c.id) + ",\"frame\":" + std::to_string(c.frame) +
                    ",\"image\":" + (c.png.empty() ? "false" : "true") + ",\"pick\":" +
                    (c.picked ? "[" + std::to_string(c.pick_x) + "," + std::to_string(c.pick_y) + "]" : "null") +
                    ",\"passes\":[";
    for (size_t i = 0; i < c.passes.size(); i++) {
        const Pass& p = c.passes[i];
        if (i) o += ',';
        o += "{\"label\":";
        str(o, p.label);
        o += ",\"colors\":[";
        for (size_t j = 0; j < p.colors.size(); j++) o += (j ? "," : "") + id(p.colors[j]);
        o += "],\"depth\":" + id(p.depth) + ",\"width\":" + std::to_string(p.width) +
             ",\"height\":" + std::to_string(p.height) + '}';
    }
    o += "],\"textures\":{";
    bool first = true;
    for (auto& [k, t] : c.textures) {
        if (!first) o += ',';
        first = false;
        o += id(k) + ":{\"name\":";
        str(o, t.name);
        o += ",\"format\":";
        str(o, t.format);
        o += ",\"width\":" + std::to_string(t.width) + ",\"height\":" + std::to_string(t.height) +
             ",\"mips\":" + std::to_string(t.mips) + ",\"layers\":" + std::to_string(t.layers) + '}';
    }
    // programs once each; draws refer to them by id
    std::map<uint64_t, const Program*> programs;
    for (auto& d : c.draws)
        if (d.program) programs[d.program->id] = d.program.get();
    o += "},\"programs\":{";
    first = true;
    for (auto& [k, p] : programs) {
        if (!first) o += ',';
        first = false;
        o += id(k) + ":{\"name\":";
        str(o, p->name);
        o += ",\"shaders\":[";
        for (size_t j = 0; j < p->shaders.size(); j++) {
            const Shader& s = p->shaders[j];
            if (j) o += ',';
            o += "{\"stage\":";
            str(o, s.stage);
            o += ",\"name\":";
            str(o, s.name);
            char h[24];
            snprintf(h, sizeof h, "%016llx", (unsigned long long)s.hash);
            o += ",\"hash\":\"" + std::string(h) + "\",\"size\":" + std::to_string(s.size) + '}';
        }
        o += "],\"state\":";
        kvs(o, p->state);
        o += '}';
    }
    o += "},\"draws\":[";
    for (size_t i = 0; i < c.draws.size(); i++) {
        const Draw& d = c.draws[i];
        if (i) o += ',';
        o += "{\"kind\":";
        str(o, d.kind);
        o += ",\"key\":" + id(d.key) + ",\"samples\":" + std::to_string(d.samples) + ",\"count\":" + std::to_string(d.count) + ",\"instances\":" + std::to_string(d.instances) +
             ",\"pass\":" + std::to_string(d.pass) + ",\"label\":";
        str(o, d.label);
        o += ",\"program\":" + (d.program ? id(d.program->id) : std::string("null")) + ",\"dynamic\":";
        kvs(o, d.dynamic);
        o += ",\"textures\":";
        kvs(o, d.textures);
        o += '}';
    }
    return o + "]}";
}

// Hands a request to the backend and waits for its result (a few frames). False on a timeout or when another
// request is still running; res then holds the error.
bool run(const Request& r, httplib::Response& res, std::unique_lock<std::mutex>& lock) {
    if (g_pending || g_taken) {  // taken: the backend is still on an earlier one (even one that timed out)
        res.status = 409;
        res.set_content("{\"error\":\"another request is still running\"}", "application/json");
        return false;
    }
    uint64_t before = g_done;
    g_request = r, g_pending = true;
    if (!g_published.wait_for(lock, std::chrono::seconds(15), [&] { return g_done != before; })) {
        g_pending = false;
        res.status = 504;
        res.set_content("{\"error\":\"the game presented no frames for 15 s\"}", "application/json");
        return false;
    }
    return true;
}

void finished() {  // g_mutex held
    g_pending = false;
    g_done++;
    g_published.notify_all();
}

std::shared_ptr<const Capture> find_locked(int capture_id) {
    for (auto& c : g_captures)
        if ((int)c->id == capture_id) return c;
    return nullptr;
}

std::shared_ptr<const Capture> find(int capture_id) {
    std::lock_guard lock(g_mutex);
    for (auto& c : g_captures)
        if ((int)c->id == capture_id) return c;
    return nullptr;
}

void serve() {
    static httplib::Server server;
    int port = 7700;
    if (const char* e = getenv("FIXFRAME_PORT")) port = atoi(e);
    int last = port + 20;  // the next free one: every Vulkan process with the layer starts a server
    while (port < last && !server.bind_to_port("127.0.0.1", port)) port++;
    if (port == last) {
        fprintf(stderr, "[fixframe] no free port for the UI\n");
        return;
    }
    // Any web page can reach localhost: answer only our own page (Origin) at our own name (Host, against
    // DNS rebinding).
    const std::string a = "127.0.0.1:" + std::to_string(port), b = "localhost:" + std::to_string(port);
    server.set_pre_routing_handler([a, b](const httplib::Request& req, httplib::Response& res) {
        std::string host = req.get_header_value("Host"), origin = req.get_header_value("Origin");
        bool ok = (host == a || host == b) && (origin.empty() || origin == "http://" + a || origin == "http://" + b);
        if (ok) return httplib::Server::HandlerResponse::Unhandled;
        res.status = 403;
        return httplib::Server::HandlerResponse::Handled;
    });

    for (const Asset& asset : kAssets) {
        std::string path = asset.path == std::string("index.html") ? "/" : "/" + std::string(asset.path);
        server.Get(path, [&asset](const httplib::Request&, httplib::Response& res) {
            res.set_content(reinterpret_cast<const char*>(asset.data), asset.size, asset.mime);
        });
    }
    server.Get("/api/status", [](const httplib::Request&, httplib::Response& res) {
        std::lock_guard lock(g_mutex);
        std::string o = "{\"app\":";
        str(o, g_app);
        o += ",\"device\":";
        str(o, g_device);
        o += ",\"backend\":";
        str(o, g_backend);
        o += ",\"frame\":" + std::to_string(g_frame) + ",\"draws\":" + std::to_string(g_draws) +
             ",\"fps\":" + std::to_string(g_times.size()) + ",\"captures\":[";
        for (size_t i = 0; i < g_captures.size(); i++)
            o += (i ? "," : "") + std::string("{\"id\":") + std::to_string(g_captures[i]->id) + ",\"frame\":" +
                 std::to_string(g_captures[i]->frame) + ",\"draws\":" + std::to_string(g_captures[i]->draws.size()) + '}';
        res.set_content(o + "]}", "application/json");
    });
    server.Post("/api/capture", [](const httplib::Request&, httplib::Response& res) {
        std::unique_lock lock(g_mutex);
        if (run({Request::kCapture}, res, lock))
            res.set_content("{\"id\":" + std::to_string(g_result_capture) + '}', "application/json");
    });
    // A pick: a new capture whose draws carry the samples they wrote at x, y (0..1 of the image).
    server.Post("/api/pick", [](const httplib::Request& req, httplib::Response& res) {
        Request r{Request::kPick};
        r.x = std::stof(req.get_param_value("x").empty() ? "0" : req.get_param_value("x"));
        r.y = std::stof(req.get_param_value("y").empty() ? "0" : req.get_param_value("y"));
        std::unique_lock lock(g_mutex);
        if (run(r, res, lock))
            res.set_content("{\"id\":" + std::to_string(g_result_capture) + '}', "application/json");
    });
    // A scrub: the target of a capture's draw as it is right after that draw, from a new frame.
    server.Post(R"(/api/captures/(\d+)/draws/(\d+)/target)", [](const httplib::Request& req, httplib::Response& res) {
        std::unique_lock lock(g_mutex);
        auto c = find_locked(std::stoi(req.matches[1]));
        size_t i = std::stoul(req.matches[2]);
        if (!c || i >= c->draws.size()) { res.status = 404; return; }
        std::map<uint64_t, int> seen;
        for (const Draw& d : c->draws) seen[d.key]++;
        size_t anchor = i;
        while (anchor > 0 && seen[c->draws[anchor].key] != 1) anchor--;
        if (seen[c->draws[anchor].key] != 1) {
            res.status = 422;
            res.set_content("{\"error\":\"Nothing before this draw is unique enough in the frame to find it again.\"}",
                            "application/json");
            return;
        }
        Request r{Request::kScrub};
        r.anchor = c->draws[anchor].key, r.target = c->draws[i].key;
        for (size_t j = anchor + 1; j < i; j++) r.nth += c->draws[j].key == r.target;
        if (!run(r, res, lock)) return;
        if (g_result_png.empty()) {
            res.status = 422;
            std::string o = "{\"error\":";
            str(o, g_result_error);
            res.set_content(o + '}', "application/json");
            return;
        }
        res.set_content(g_result_png, "image/png");
    });
    server.Get(R"(/api/captures/(\d+))", [](const httplib::Request& req, httplib::Response& res) {
        auto c = find(std::stoi(req.matches[1]));
        if (!c) { res.status = 404; return; }
        res.set_content(capture_json(*c), "application/json");
    });
    server.Get(R"(/api/captures/(\d+)/image.png)", [](const httplib::Request& req, httplib::Response& res) {
        auto c = find(std::stoi(req.matches[1]));
        if (!c || c->png.empty()) { res.status = 404; return; }
        res.set_content(c->png, "image/png");
    });

    fprintf(stderr, "[fixframe] UI: http://127.0.0.1:%d\n", port);
    server.listen_after_bind();
}

}  // namespace

void start(const std::string& app, const std::string& device, const std::string& backend) {
    static std::once_flag once;
    std::call_once(once, [&] {
        {
            std::lock_guard lock(g_mutex);
            g_app = app, g_device = device, g_backend = backend;
        }
        std::thread(serve).detach();
    });
}

void frame_done(uint64_t frame, uint32_t draws) {
    std::lock_guard lock(g_mutex);
    g_frame = frame, g_draws = draws;
    double t = now();
    g_times.push_back(t);
    while (g_times.front() < t - 1.0) g_times.pop_front();
}

bool take_request(Request& out) {
    std::lock_guard lock(g_mutex);
    if (!g_pending || g_taken) return false;
    g_taken = true;
    out = g_request;
    return true;
}

void publish(std::unique_ptr<Capture> capture) {
    std::lock_guard lock(g_mutex);
    capture->id = g_next_id++;
    g_result_capture = capture->id;
    g_captures.push_back(std::move(capture));
    if (g_captures.size() > kKeep) g_captures.pop_front();
    g_taken = false;
    finished();
}

void scrubbed(std::string png, std::string error) {
    std::lock_guard lock(g_mutex);
    g_result_png = std::move(png), g_result_error = std::move(error);
    g_taken = false;
    finished();
}

}  // namespace fixframe::core
