# CLAUDE.md – Villa Tennelbach (Villa Callenberg) 3D reconstruction

Handover notes for continuing this project in a later session (claude.ai or Claude Code).
State as of **4 Oct 2026**. Conversation language with Moritz: German or English (answer in the language he uses).

## 1. What this is

3D reconstruction of a demolished villa in the Tennelbach valley, Wiesbaden-Sonnenberg, designed in **September 1904**
for "Ihre Excellenz Frau General Callenberg" (plans signed "Diez im September 1904", signature of the
"Bauleitender" not legible; *Diez* is the place of signing, not the architect).
Deliverables: a GLB model, a self-contained three.js web viewer, a path-traced render.

- Repo: https://github.com/moritzramm/tennelbach (public). A commit `5af3732` with viewer, model, standalone
  version, tools and README was prepared on top of `763bf9d Initial commit`; Moritz pushes himself (Claude has no
  GitHub credentials – do not ask for tokens in chat).
- GitHub Pages (main, root): https://moritzramm.github.io/tennelbach/
- Claude artifact of the viewer (private until shared): https://claude.ai/artifact/LKFdXJRY7V5eERmd8WgCRi

## 2. Repository layout

| Path | Purpose |
|---|---|
| `index.html` | Viewer; three.js r128 + GLTFLoader + OrbitControls inlined; **fetches** `villa_callenberg.glb` |
| `villa_callenberg.glb` | Model (glTF binary, Y-up, metres, ~147k triangles, 14 material groups) |
| `standalone/villa_callenberg_viewer.html` | Same viewer with GLB embedded as base64 (works via file://) |
| `preview.jpg` | Cycles render (trees/lawn are invented placeholders) |
| `tools/build.py` | Generates the GLB (all geometry + materials) |
| `tools/overlay.py` | Orthographic raster of the model aligned pixel-accurately on the elevation scans (verification) |
| `tools/render.py` | Quick matplotlib previews (painter's algorithm – visual artifacts are NOT geometry errors) |
| `tools/photoreal_render.py` | Blender 5 / bpy Cycles render |

Source scans (not in repo; rights of photos unclear): `A0_sw.jpg` (cellar + attic plan), `B0_sw.jpg` (ground + upper
floor plan), `D0_sw.jpg` (front + right elevation), `E0_sw.jpg` (left + back elevation), a b/w photo of the
front-left (as built), a colour photo of the left side during demolition, another b/w photo of the right side.
`build.py` and `overlay.py` expect the scans at `/mnt/user-data/uploads/` – adjust paths if needed.

## 3. Rebuild

```
pip install trimesh shapely manifold3d scipy mapbox_earcut opencv-python-headless
python tools/build.py            # -> villa_callenberg.glb (+ geo.pkl for overlay/render)
python tools/overlay.py front right left back   # -> ov_<view>.png, drawing in red over model
```
After rebuilding: copy GLB next to `index.html`; for the standalone/artifact viewer replace the base64 string in
`const B64="…"` (keep everything else).

Photoreal render (Python 3.12 has no bpy wheel):
```
pip install uv && uv venv /tmp/bl --python 3.11 && uv pip install --python /tmp/bl/bin/python bpy   # Blender 5.0.1
/tmp/bl/bin/python tools/photoreal_render.py -- 1000 64 out.png   # ~4 min on 1 CPU core
```
Camera of the existing render was fitted to a viewer screenshot (back-left view).

## 4. Coordinates and calibration

- Plan coords: **x** left→right (as in front elevation), **y** front (0) → back (14.5), **z** up, metres.
- GLB: Y-up, centred: `gltf = (x-5.74, z, -(y-7.25))`. Blender import: `blender = plan - (5.74, 7.25, 0)`.
- Footprint: 11.48 × 14.50; open corner ("Nische") front-right x 9.95–11.48, y 0–3.2 (terrace over cellar, open to
  sky); back-left recess 0.57 (x 0–4.72 ends at y 13.93); ground-floor chamfer (10.85, 3.2)→(11.48, 3.85);
  Salon bay: straight cheeks to y −0.53, segmental arc centre (2.62, 0.48), R 1.75, depth 1.27, width 2.86 (x 1.19–4.05).
- Terrain `zt(y)`: 0.1 + 1.7·(y/14.5)² between y 0 and 14.5 (slopes up to the back), linear outside, clamped.
- Elevation calibration (scale bar 94.6 px/m on both sheets; front/right drawn ~1.8 % narrower than the plans):
  `px = X0 + sx·u`, `py = Y0 − 94.6·z`

  | View | Sheet | X0 | sx | Y0 | u |
  |---|---|---|---|---|---|
  | front | D0 | 242 | 92.9 | 2178.7 | x |
  | right | D0 | 1989 | 92.9 | 2178.7 | y |
  | left | E0 | 328 | 94.5 | 2238.7 | 14.5 − y |
  | back | E0 | 2351 | 93.7 | 2238.7 | 11.48 − x |

## 5. Model state (key dimensions)

- Heights: plinth top 2.18 (rusticated ashlar with staggered joints), string course 6.0–6.3 (front, right and
  corner walls only), architrave 9.98–10.12, frieze 10.12–10.5, main cornice top **ZC = 10.82**.
- Roofs (frustums, min-of-planes, unioned with manifold):
  - Main: base (−0.5, −0.62)–(10.45, 15.0), flat top (4.9, 4.6)–(5.6, 9.0) at z 17.0 with railing and two finials
  - Right wing incl. tower: base (4.1, 2.7)–(11.98, 15.0), ridge (7.95…8.1, 6.35…12.0) at z 15.2
- Gables = **decorative gables (Ziergiebel) in front of straight saddle roofs** (measured half-profiles):
  front centre x 7.3, crown 15.12, ridge 14.06; right (ogee) centre y 7.65, top 14.36, ridge 13.76;
  left (mansard) centre y 6.78, top 13.43, ridge 14.26 (roof end forms the "hood").
- Dormers: bell hood only as a short front piece + saddle roof behind; back dormers and back-left Zwerchhaus have
  saddle roofs (no barrel vaults anywhere).
- Chimneys (per attic plan): (7.45, 5.65) 0.85×0.7 top 17.2; (2.65, 9.75) 1.2×0.7 top 16.4.
- Windows: modelled individually from the elevations (`win()` in build.py: recess, frame, sashes, transom, glazing
  bars, profiled 3-step surround with ears/feet, sills, hoods, keystones). Ornaments are vectorised from the
  elevation linework (`trace()` + `relief()`), incl. wrought iron of the veranda.
- Tower: ground-floor windows, upper-floor loggia with columns/arches/canopy; balcony door on the loggia's inner
  wall (x = 10.1).
- Colours (from the colour photo, sky-white-balanced): plaster `#CDBC98`, cornices/surrounds `#B2A280` (~20 % darker),
  plinth `#787460`, slate `#313E4E`, sashes white `#EDEBE3`, zinc, iron `#222224`.

## 6. Decisions made with Moritz (do not revert without asking)

- Tall arched opening on the left facade is a **blind arch** (Blendarkade), not a window.
- Ground-floor Salon window on the left side is a **blind window** (closed in drawing and colour photo).
- Projecting trim (surrounds, cornices) slightly **darker** than plaster.
- **No roof over the open front-right corner and no corner pier** (both appear in the 1904 elevations but not in
  plans/photos – removed on request).
- **No curved roof bodies**: gables and dormers are decorative fronts with straight saddle roofs behind.
- Viewer background always **dark** (`data-theme="dark"`, `#1f1c18`).

## 7. Viewer features and implementation notes

- Presets (Perspektive, Vorder-/Hinteransicht, Seiten), terrain toggle, auto-rotation.
- **Licht** panel: one directional light, brightness 0–300 % (default 125), direction 0–359° (default 323, 0° = front),
  height 2–88° (default 36); shadows follow.
- **Kamerafahrt** (one-shot): `TOUR = {h1: 8.0, h2: 37.0, ty: 8.5, dur: 30000, pre: 1800, total: 540, rise: 0}`;
  starts at upper-floor height, rises during the whole 540° orbit to ~20 m above the roof, always looking at the house
  centre; radius = fit-to-aspect × 1.75; same direction as "Rotation". UI (bar, light panel, hint) hidden via
  `body.touring`; **Esc** (or tap on touch devices) stops; UI returns at stop/end.
- WebGL failure → automatic Canvas-2D software renderer (all features work, slower). Moritz's environment once had no
  WebGL, so keep the fallback.
- Error overlay via `showErr()`; `index.html` explains if opened via file://.

## 8. Open points / ideas

- Roof top: b/w photo suggests a normal ridge with two small finials instead of the drawn flat platform with railing
  (asked, not yet decided).
- As built (b/w photo) the front gable was a simple triangle and the front dormer rectangular – model follows 1904.
- Window sashes appear dark green-grey in the colour photo (model: white).
- Back upper-floor window is drawn without glazing bars (maybe blind) – model: glazed.
- Terrain only ~±16 m around the house; edges visible during the camera flight.
- Render: trees are placeholder blobs; possible improvements or other views/times of day.
- No license chosen for the repo yet.
- History research leads: Wiesbaden address books 1839–1948 digitised by HLB RheinMain (vol. 1904/05; 1907 incl.
  Sonnenberg); Stadtarchiv Wiesbaden building file (Bauakte); Sonnenberg independent until 28 Oct 1926;
  identify "General Callenberg" via address book + Prussian army lists.

## 9. Pitfalls learned

- `win(..., gd=…)`: glass depth must be **positive** (a negative value placed the loggia door 1.95 m outside the wall).
- `trace()` picks up roof hatching/background lines – pass a `mask` polygon (gable crown, dormer hood, veranda brackets).
- Cornice rings must follow the building outline `P`, not `P_full` (otherwise beams float over the open corner).
- `dormer()` uses `z_roof()` → `ROOFS` must be final before dormers are placed; gables add `ROOF_CUT` boxes that
  trim the main-roof overhang in front of them.
- Boolean ops (manifold) need watertight inputs (extrusions, boxes, convex hulls are fine).
- matplotlib previews show painter-sorting artifacts; verify geometry with `overlay.py` or the WebGL viewer.
- Background processes in the sandbox get killed when the tool call ends – use `setsid nohup … &` and poll.
