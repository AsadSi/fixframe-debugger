# Fixframe

Live render debugger for emulators and ports. It sits inside a running game's renderer and lets you find
and fix a rendering bug while the game runs.

- **Vulkan layer**: hooks GPU work with no change to the game (emulators, ports; macOS/MoltenVK and Linux).
- **C SDK** (`sdk/fixframe_sdk.h`): for custom renderers a layer can't see (used with a Metal renderer).
- **Web UI and CLI**: list a frame's draws, click a pixel to find the draw behind it, scrub the frame to
  draw N, compare two draws, and save a capture (`.ffcap`) that opens without the game.

Where RenderDoc captures a frame and lets you study it, Fixframe works on the live game: you change a
draw and see the result at once. Use both. Credits: [CREDITS.md](CREDITS.md).

Status: early. The layer captures, picks and scrubs; overrides, shader editing and rule export are in
progress. Expect rough edges.

## Build and run

```
cmake -S . -B build -G Ninja && cmake --build build
VK_LAYER_PATH=$PWD/build VK_INSTANCE_LAYERS=VK_LAYER_FIXFRAME_debug <your Vulkan app>
```

Log lines start with `[fixframe]` on stderr; the UI is at http://127.0.0.1:7700 (next free port per
process). A saved capture opens with no game: `cli/fixframe open x.ffcap`.

## Third-party code

Vendored under `third_party/`, each with its source and commit in `SOURCE`: vkroots, Vulkan-Utility-Libraries,
cpp-httplib, nlohmann/json, stb. See each folder's LICENSE.

MIT licence, see LICENSE.

## More projects

- [Wind Waker HD: native port + mods](https://github.com/AsadSi/wwhd-recomp-mods): spooky sea events, Tetra's ship, online co-op
- [Twilight Princess HD: Recompiled](https://github.com/AsadSi/twilight-princess-hd-recomp): native macOS port of the Wii U version
- [Ocarina of Time: Recompiled](https://github.com/AsadSi/ocarina-of-time-recomp): native port with mods (unofficial fork)
