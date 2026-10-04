# tools

| Script | Purpose |
|---|---|
| `build.py` | Generates `villa_callenberg.glb` (trimesh, shapely, manifold3d, scipy, mapbox_earcut, opencv-python-headless). |
| `overlay.py` | Rasterises the model orthographically and overlays it pixel-accurately on the elevation drawings. |
| `render.py` | Quick matplotlib preview renders. |
| `photoreal_render.py` | Path-traced render with Blender 5 / `bpy` (Cycles). |

`build.py` traces ornaments from the elevation scans `D0_sw.jpg` and `E0_sw.jpg`, which are expected under
`/mnt/user-data/uploads/`. These scans are not part of this repository; adjust the paths in the scripts if you have them.

    pip install trimesh shapely manifold3d scipy mapbox_earcut opencv-python-headless
    python build.py
