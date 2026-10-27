# Fixframe

Live render debugger for emulators and ports. It sits inside a running game's renderer and shows you what
the frame on screen is made of, while the game runs.

- **Vulkan layer**: hooks GPU work with no change to the game (emulators, ports). One universal macOS
  library, so native arm64 and Rosetta x86_64 processes both load it.
- **C++ SDK** (`sdk/fixframe_sdk.h`): for renderers a layer can't see (Metal, a custom backend). The
  renderer reports its draws, including the ones it skipped and why.
- **Web UI and CLI**: served by the game process on `127.0.0.1`; a saved capture opens without the game.

Where RenderDoc captures a frame for you to study, Fixframe works on the live game. Use both.
Credits: [CREDITS.md](CREDITS.md).

## Status: 0.1, an early preview

What works today:

| | Vulkan layer | SDK |
|---|---|---|
| Capture a frame: passes, draws, pipelines/programs, shaders (names, hashes), state, textures bound | yes | yes |
| Draws the renderer skipped, with the reason | | yes |
| Click a pixel to find the draws that wrote it (occlusion queries) | yes | not yet |
| Scrub: a draw's target right after that draw | yes | not yet |
| Compare two draws, of the same or different captures | yes | yes |
| Save a capture (`.ffcap`) and open it with the game closed | yes | yes |

Tested on macOS (Apple Silicon, MoltenVK): the layer in a Rosetta Vulkan port and in its own smoke test,
the SDK in a Wind Waker HD recomp's Metal renderer. Linux should build but hasn't been tested yet.

Not built yet (see the roadmap below): changing a draw live (overrides), shader view and edit, a
rolling buffer for bugs that last a few frames, and a CLI for scripted checks.

## Build

Needs CMake 3.24+, Ninja, a C++20 compiler and the Vulkan headers (and the Vulkan loader for the smoke
test). Everything else is vendored under `third_party/`.

```
cmake -S . -B build -G Ninja
cmake --build build
```

## Use it with a Vulkan game

```
VK_LAYER_PATH=$PWD/build VK_INSTANCE_LAYERS=VK_LAYER_FIXFRAME_debug <your Vulkan app>
```

Lines from Fixframe start with `[fixframe]` on stderr, including where the UI is:
`http://127.0.0.1:7700` (set `FIXFRAME_PORT` to choose; each process takes the next free port). Try it
without a game first:

```
VK_LAYER_PATH=build VK_INSTANCE_LAYERS=VK_LAYER_FIXFRAME_debug build/fixframe_smoke
```

In the UI: **Capture frame**, then click a draw for everything it used, click the picture to find the
draws behind a pixel, **Show its target right after this draw** to scrub, **Compare other draws with this one** to diff two draws,
and **Save capture** for a `.ffcap` file.

## Use the SDK in your own renderer

Add Fixframe to your build and link the static library `fixframe_sdk`:

```cmake
add_subdirectory(path/to/fixframe fixframe)
target_link_libraries(your_renderer PRIVATE fixframe_sdk)
```

Then report from the render thread (`sdk/fixframe_sdk.h` has the details):

```cpp
#include "fixframe_sdk.h"

fixframe::sdk::start("My game", "Apple M4", "Metal");          // once, when the renderer is up

if (fixframe::sdk::capturing()) {                               // per draw: cheap when not capturing
    fixframe::Draw d;
    d.kind = "indexed";
    d.count = index_count;
    d.label = "sea";
    d.pass = fixframe::sdk::pass({"main", {color_target_id}, depth_id, width, height});
    if (!pipeline_ok) d.skipped = "shader-failed";              // report skipped draws too
    fixframe::sdk::capture()->draws.push_back(d);
}

fixframe::sdk::frame_end(frame, draw_count, [] { return png_of_screen(); });  // once per frame
```

`fixframe::core::png(rgba, width, height)` turns RGBA8 pixels into the PNG `frame_end` asks for.

## Saved captures

```
cli/fixframe open capture.ffcap
```

Python 3 only, no packages. The capture opens in the same UI. Picking and scrubbing need the game
running, so they are off.

## Safety

The UI server binds to `127.0.0.1` only and answers only its own page: any web page can reach
localhost, so requests with another `Host` or `Origin` are refused. Nothing is sent anywhere else.

## Roadmap

1. Exact picks everywhere (target filter, static scissor, which draws write a target) and a fuller
   inspector.
2. A rolling buffer of the last seconds, a freeze trigger, a timeline and plotted values, for bugs that
   last a frame or two.
3. Overrides on the live game: skip a draw, change its state, blend, constants or textures; save them as
   rules.
4. Shader view (SPIRV-Cross) and edit with hot-reload; a texture viewer.
5. A CLI for scripts and tests (capture, pick, assert a pixel or an image, diff).
6. Regression runs on a game's replay, Linux checks, docs and a tutorial.

Issues and pull requests are welcome.

## Licence

MIT, see [LICENSE](LICENSE). Third-party code under `third_party/` keeps its own licence; each folder
has its LICENSE and a `SOURCE` file with its origin and commit.

## More projects

- [Wind Waker HD: native port + mods](https://github.com/AsadSi/wwhd-recomp-mods): spooky sea events, Tetra's ship, online co-op
- [Twilight Princess HD: Recompiled](https://github.com/AsadSi/twilight-princess-hd-recomp): native macOS port of the Wii U version
- [Ocarina of Time: Recompiled](https://github.com/AsadSi/ocarina-of-time-recomp): native port with mods (unofficial fork)
