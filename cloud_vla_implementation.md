# Cloud-Based VLA Robot Control — Implementation Design

## Executive Summary

This document defines the implementation plan for cloud-based VLA (Vision-Language-Action)
inference integrated with the Tashan + Isaac Sim robotic setup. It covers platform selection,
architecture, protocol design, server implementations, and cost analysis.

Based on analysis of the existing codebase (4 VLA scenarios, Colab server, Tashan TS-F-A
sensor integration) and current cloud GPU market research (March 2026).

---

## 1. Platform Recommendation (Ranked)

| Rank | Platform | Model | Cost/hr | Latency | Free Tier | Best For |
|------|----------|-------|---------|---------|-----------|----------|
| **1** | **Modal.com** | Octo-Small | $0.59 (T4) | 80–150 ms | $30/mo credits | Research & prototyping |
| **2** | **HF Inference Endpoints** | Octo-Small | $0.50 (T4) | 90–180 ms | None | Always-on demos |
| **3** | **GCP Cloud Run + GPU** | Octo-Small | $0.67 (L4) | 70–120 ms | $300 trial | Production |
| **4** | **Modal.com** | OpenVLA-7B | $1.10 (A10G) | 230–250 ms | $30/mo credits | Advanced generalization |
| **5** | **Google Colab** *(existing)* | Octo-Small | $0 | 200–500 ms | Free (12h limit) | Zero-cost testing |

### Why Modal.com is Recommended (#1)

- **Scale to zero** — pay nothing when idle (unlike HF Endpoints which charge while running)
- **$30/mo free credits** — ~50 hours of T4 GPU, enough for substantial research
- **<2 s cold starts** — much faster than Replicate (10–30 s) or GCP Cloud Run (~5 s)
- **Pure Python deployment** — no Docker, no YAML, just `@modal.cls(gpu="T4")`
- **Warm latency 80–150 ms** — well within the <500 ms requirement, close to ideal <200 ms

### Notable Model: SmolVLA (450M params)

**SmolVLA** (`lerobot/smolvla_base`) is a 450M-parameter VLA model trained on 10M frames
from 487 datasets. It runs on consumer hardware with latency comparable to Octo but
potentially better generalization. Worth considering alongside Octo.

### Platforms NOT Recommended

| Platform | Reason |
|----------|--------|
| **HF Free Inference API** | No VLA/robotics models available on serverless tier |
| **Replicate.com** | 10–30 s cold starts; no pre-deployed VLA models |
| **GCP Vertex AI** | Always-on billing (no scale to zero); higher complexity |
| **GKE with GPU** | Highest operational complexity; overkill for research |

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LOCAL (Isaac Sim + Tashan TS-F-A)                                      │
│  scenario_tashan_cloud_vla.py                                           │
│                                                                          │
│  Camera 256x256 ──┐                                                      │
│  EE state ────────┤── JPEG+JSON ──► HTTPS POST ──► ┌───────────────────┐│
│  Language inst ───┘     (~15 KB)      (pooled)      │ CLOUD BACKEND     ││
│                                                      │                   ││
│  ◄── JSON action chunk ◄── HTTPS 200 ◄──────────── │ Modal / HF / GCP  ││
│       (1, H, 7)              (~1 KB)                 │ Octo VLA on GPU   ││
│                                                      └───────────────────┘│
│  Jacobian IK ──► Joint targets ──► Franka Panda                          │
│  Tashan sensor ──► 11-ch data ──► Plots + Rerun                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key change from existing Colab approach:** REST (HTTPS) instead of WebSocket + ngrok.
This eliminates:
- Manual ngrok URL copying
- Colab session timeouts
- TCP tunnel instability

---

## 3. File Plan

### New files to create

```
ts_tactile_extension_python/
├── scenario_tashan_cloud_vla.py          # Isaac Sim scenario (client)
├── cloud_vla_server/
│   ├── modal_app.py                      # Modal.com serverless deployment
│   ├── hf_handler.py                     # HuggingFace Inference Endpoint handler
│   ├── cloudrun_app.py                   # GCP Cloud Run Flask server
│   ├── Dockerfile                        # For GCP Cloud Run
│   └── requirements.txt                  # Server-side dependencies
```

### Modified files

```
├── ui_builder.py                         # Add import option for CloudVLAScenario
```

---

## 4. Unified REST Protocol

All three server implementations share the same API, so switching backends is a
**one-line config change** (just change `CLOUD_BACKEND` and the URL).

### Health Check

```
GET /health

Response:
{
  "status": "ok",
  "model": "octo-small-1.5",
  "gpu": "T4",
  "action_horizon": 4
}
```

### Inference

```
POST /predict
Content-Type: application/json

Request:
{
  "image_b64": "<base64 JPEG 256x256>",       // ~15 KB compressed
  "ee_pos": [0.3, 0.0, 0.2],                  // end-effector position (3,)
  "ee_rot": [0.0, 0.0, 0.0, 1.0],             // end-effector quaternion xyzw (4,)
  "gripper_qpos": [0.04, 0.04],               // finger joint positions (2,)
  "language": "pick up the blue cube",
  "reset": false                                // true to clear image history
}

Response:
{
  "status": "ok",
  "action": {
    "end_effector_position": [[[dx,dy,dz], ...]],    // (1, H, 3)
    "end_effector_rotation": [[[rx,ry,rz], ...]],    // (1, H, 3) euler
    "gripper_close": [[[g], ...]]                     // (1, H, 1)
  },
  "inference_ms": 62.3,
  "model": "octo-small-1.5"
}
```

### Error Response

```json
{
  "status": "error",
  "message": "Description of what went wrong"
}
```

---

## 5. Scenario Client Design — `scenario_tashan_cloud_vla.py`

Follows the identical scenario class API used by all existing scenarios.

### Configuration Block

```python
# ======================================================================
# Cloud VLA Configuration
# ======================================================================
CLOUD_BACKEND = "modal"  # "modal" | "hf-endpoint" | "gcp-cloudrun" | "colab"

# Modal.com (recommended — $0.59/hr T4, scale-to-zero, $30/mo free)
MODAL_ENDPOINT_URL = ""  # Set after: modal deploy cloud_vla_server/modal_app.py

# HuggingFace Inference Endpoint (~$0.50/hr T4, always-on)
HF_ENDPOINT_URL = ""     # Set after creating endpoint on huggingface.co
HF_API_TOKEN = ""        # Or set HF_TOKEN environment variable

# GCP Cloud Run (~$0.67/hr L4, scale-to-zero)
GCR_ENDPOINT_URL = ""    # Set after: gcloud run deploy

# Colab fallback (free, existing infrastructure)
COLAB_SERVER_URL = "ws://0.tcp.ngrok.io:12345"

# Shared
JPEG_QUALITY = 85
ACTION_HORIZON = 4
EXEC_STEPS = 4           # Physics steps per inference call
INFERENCE_TIMEOUT_S = 10
MAX_RETRIES = 2
```

### CloudVLAClient Class

```python
class CloudVLAClient:
    """Unified REST client for cloud VLA inference.

    Supports Modal, HuggingFace Endpoints, and GCP Cloud Run.
    Falls back to Colab WebSocket or scripted policy on failure.
    """

    def __init__(self):
        self._backend = CLOUD_BACKEND
        self._session = None       # urllib3 PoolManager for connection reuse
        self._language = "pick up the blue cube"
        self._image_history = []

    def connect(self):
        """Validate endpoint availability. Called in setup_scenario().

        HTTP GET /health -> {"status": "ok", "model": "...", "gpu": "..."}
        Connection pooling via urllib3 (available in Isaac Sim's bundled Python).
        """
        ...

    def get_action(self, rgb_256x256, ee_pos, ee_rot, gripper_qpos):
        """Send observation, receive action chunk.

        1. JPEG-compress image (~15 KB at quality=85)
        2. POST /predict with JSON payload
        3. Parse response action chunk
        4. On failure: retry once, then fall back to scripted

        Returns:
            dict with keys: end_effector_position (1,H,3),
                           end_effector_rotation (1,H,3),
                           gripper_close (1,H,1)
        """
        ...

    def set_language(self, instruction):
        """Update the language task instruction."""
        self._language = instruction

    def close(self):
        """Release HTTP connection pool."""
        ...
```

### Scenario Lifecycle

```python
class CloudVLAScenario:
    """Cloud-based VLA pick-and-place with Tashan tactile sensors.

    Identical robot setup to all other scenarios. Only the inference
    client differs — REST over HTTPS instead of ZMQ or local JAX.
    """

    def load_robot(self):
        # Identical to existing scenarios:
        # - Create Franka at /World/Franka
        # - _integrate_sensor() for left/right fingers
        # - _fix_rigid_body_transforms() — CRITICAL 1:1 scale fix
        # - _retarget_root_joint() — FixedJoint to finger
        # - _setup_collision_filtering() — sensor pads vs finger geometry
        # - Create Camera: Bridge V2 shoulder view, 256x256, 20 Hz
        #   position=[0.05, 0.40, 0.45], rotation=euler(0, 42, -28)

    def setup_scenario(self):
        # - Create 3 cubes: Blue(0.04m), Green(0.04m), Red(0.04m)
        # - Create 3 target VisualCuboids
        # - Initialize sensor RigidPrim wrappers (tactile_left, tactile_right)
        # - Create CloudVLAClient(backend=CLOUD_BACKEND)
        # - client.connect() — validate endpoint
        # - Set language instruction from UI field
        # - Initialize data buffers for plotting

    def update_scenario(self, step, step_ind):
        # Action-chunking loop (same pattern as colab/groot/lite):
        #
        # Every EXEC_STEPS physics steps:
        #   1. Capture camera RGB via get_current_frame()["rgba"][:,:,:3]
        #   2. Read EE pose from articulation (position + quaternion)
        #   3. client.get_action(rgb, ee_pos, ee_rot, gripper_qpos)
        #   4. Store action chunk buffer, reset step counter
        #   5. Record latency (inference_ms + round_trip_ms)
        #
        # Every physics step:
        #   1. Pop next action from buffer
        #   2. Jacobian IK:
        #      J = jacobians[0, ee_body_index][:, :7]   # (6, 7)
        #      JJT = J @ J.T + 0.05 * np.eye(6)
        #      joint_delta = J.T @ np.linalg.solve(JJT, ee_delta)
        #   3. Apply joint targets via ArticulationAction
        #   4. Gripper: finger_pos = (1 - gripper_close) * 0.05
        #   5. Read Tashan sensor: TSsensor(prim, range_path) → (11,)
        #   6. Log to Rerun (if available)

    def teardown_scenario(self):
        # Reset action buffer, image history, step counter
        # client.close() — release HTTP pool

    def draw_data(self):
        # Generate matplotlib plots (saved to ../sensor_data/):
        # 1. Tactile forces + proximity (*_forces.png)
        # 2. Capacitance 7 channels (*_capacitance.png)
        # 3. All 11 channels side-by-side (*_all_channels.png)
        # 4. VLA action trajectories — 6 EE dims + gripper (*_actions.png)
        # 5. Gripper vs contact force correlation (*_gripper_force.png)
        # 6. Network latency breakdown: inference_ms vs round_trip_ms (*_latency.png)
        # 7. Latency timeline: stacked area chart (*_latency_timeline.png)
```

### Automatic Backend Fallback

```python
BACKEND_PRIORITY = ["modal", "hf-endpoint", "gcp-cloudrun", "colab", "scripted"]

def get_action(self, obs):
    for backend in BACKEND_PRIORITY:
        try:
            return self._clients[backend].predict(obs)
        except (ConnectionError, TimeoutError):
            print(f"[CloudVLA] {backend} failed, trying next...")
    return self._scripted_policy.get_action(obs)  # zero-cost fallback
```

---

## 6. Server Implementations

### 6.1 Modal.com — `cloud_vla_server/modal_app.py`

```python
import modal

app = modal.App("tashan-octo-vla")

vla_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "jax[cuda12_pip]",
        "git+https://github.com/kvablack/dlimp.git",
        "git+https://github.com/octo-models/octo.git",
        "Pillow", "numpy", "flask",
    )
)

@app.cls(
    gpu="T4",                        # $0.59/hr, 16 GB VRAM (Octo needs ~2 GB)
    image=vla_image,
    container_idle_timeout=300,      # Keep warm 5 min after last request
    allow_concurrent_inputs=1,       # One inference at a time (stateful history)
)
class OctoVLA:
    @modal.enter()
    def load_model(self):
        """Runs once when container starts. Loads model into GPU."""
        import jax
        if not hasattr(jax, 'tree_map'):
            jax.tree_map = jax.tree_util.tree_map
        from octo.model.octo_model import OctoModel

        self.model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
        self.task = None
        self.image_history = []
        self.stats = self.model.dataset_statistics["bridge_dataset"]["action"]
        print(f"[Modal] Octo loaded on {jax.devices()}")

    @modal.web_endpoint(method="GET")
    def health(self):
        return {"status": "ok", "model": "octo-small-1.5", "gpu": "T4"}

    @modal.web_endpoint(method="POST")
    def predict(self, data: dict):
        """Receive observation, return action chunk."""
        import time, base64, io, numpy as np, jax
        from PIL import Image

        if data.get("reset", False):
            self.image_history = []
            self.task = None
            return {"status": "ok"}

        # Set/update language task
        lang = data.get("language", "pick up the object")
        if self.task is None:
            self.task = self.model.create_tasks(texts=[lang])

        # Decode image
        img_bytes = base64.b64decode(data["image_b64"])
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        if img.shape[:2] != (256, 256):
            img = np.array(Image.fromarray(img).resize((256, 256)))

        # History management (window=2)
        self.image_history.append(img)
        if len(self.image_history) > 2:
            self.image_history = self.image_history[-2:]

        # Pad to window size
        history = list(self.image_history)
        while len(history) < 2:
            history.insert(0, history[0])

        images = np.stack(history)[np.newaxis]  # (1, 2, 256, 256, 3)
        pad_mask = np.zeros((1, 2), dtype=bool)
        pad_mask[0, -len(self.image_history):] = True

        # Inference
        t0 = time.time()
        actions = self.model.sample_actions(
            {"image_primary": images, "pad_mask": pad_mask},
            self.task,
            unnormalization_statistics=self.stats,
            rng=jax.random.PRNGKey(int(t0 * 1000) & 0xFFFFFFFF),
        )
        inference_ms = (time.time() - t0) * 1000

        # Parse: (1, H, 7) -> position(3) + rotation(3) + gripper(1)
        a = np.array(actions)
        if a.ndim == 2:
            a = a[np.newaxis]

        return {
            "status": "ok",
            "action": {
                "end_effector_position": a[:, :, :3].tolist(),
                "end_effector_rotation": a[:, :, 3:6].tolist(),
                "gripper_close": a[:, :, 6:7].tolist(),
            },
            "inference_ms": round(inference_ms, 2),
        }
```

**Deployment commands:**
```bash
pip install modal
modal setup                                    # One-time auth (browser login)
modal deploy cloud_vla_server/modal_app.py     # Deploy to Modal cloud
# Prints endpoint URLs:
#   GET  https://<user>--tashan-octo-vla-octovla-health.modal.run
#   POST https://<user>--tashan-octo-vla-octovla-predict.modal.run
```

### 6.2 HuggingFace Inference Endpoint — `cloud_vla_server/hf_handler.py`

```python
"""Custom inference handler for HuggingFace Inference Endpoints.

Deploy by creating a model repo on HuggingFace containing:
  - handler.py (this file, renamed)
  - requirements.txt

Then create an Inference Endpoint pointing to that repo.
"""

class EndpointHandler:
    def __init__(self, path=""):
        import jax
        if not hasattr(jax, 'tree_map'):
            jax.tree_map = jax.tree_util.tree_map
        from octo.model.octo_model import OctoModel

        model_id = path if path else "hf://rail-berkeley/octo-small-1.5"
        self.model = OctoModel.load_pretrained(model_id)
        self.task = None
        self.image_history = []
        self.stats = self.model.dataset_statistics["bridge_dataset"]["action"]

    def __call__(self, data):
        """Process inference request.

        Args:
            data: dict with keys matching the unified protocol
                  (image_b64, ee_pos, ee_rot, gripper_qpos, language, reset)

        Returns:
            dict with status, action chunk, and inference_ms
        """
        import time, base64, io, numpy as np, jax
        from PIL import Image

        inputs = data.get("inputs", data)  # HF wraps payload in "inputs"

        if inputs.get("reset", False):
            self.image_history = []
            self.task = None
            return {"status": "ok"}

        lang = inputs.get("language", "pick up the object")
        if self.task is None:
            self.task = self.model.create_tasks(texts=[lang])

        img_bytes = base64.b64decode(inputs["image_b64"])
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        if img.shape[:2] != (256, 256):
            img = np.array(Image.fromarray(img).resize((256, 256)))

        self.image_history.append(img)
        if len(self.image_history) > 2:
            self.image_history = self.image_history[-2:]

        history = list(self.image_history)
        while len(history) < 2:
            history.insert(0, history[0])

        images = np.stack(history)[np.newaxis]
        pad_mask = np.zeros((1, 2), dtype=bool)
        pad_mask[0, -len(self.image_history):] = True

        t0 = time.time()
        actions = self.model.sample_actions(
            {"image_primary": images, "pad_mask": pad_mask},
            self.task,
            unnormalization_statistics=self.stats,
            rng=jax.random.PRNGKey(int(t0 * 1000) & 0xFFFFFFFF),
        )
        inference_ms = (time.time() - t0) * 1000

        a = np.array(actions)
        if a.ndim == 2:
            a = a[np.newaxis]

        return {
            "status": "ok",
            "action": {
                "end_effector_position": a[:, :, :3].tolist(),
                "end_effector_rotation": a[:, :, 3:6].tolist(),
                "gripper_close": a[:, :, 6:7].tolist(),
            },
            "inference_ms": round(inference_ms, 2),
        }
```

**Deployment steps:**
1. Create a new model repo on huggingface.co (e.g., `your-user/octo-vla-endpoint`)
2. Upload `hf_handler.py` as `handler.py` and `requirements.txt`
3. Go to huggingface.co → Inference Endpoints → New Endpoint
4. Select your repo, choose T4 GPU ($0.50/hr), deploy
5. Copy the endpoint URL into `HF_ENDPOINT_URL`

### 6.3 GCP Cloud Run — `cloud_vla_server/cloudrun_app.py`

```python
"""Flask server for GCP Cloud Run with L4 GPU.

Deployed via:
  gcloud run deploy tashan-octo-vla \
    --source=cloud_vla_server/ \
    --gpu=1 --gpu-type=nvidia-l4 \
    --memory=16Gi --cpu=4 \
    --region=us-central1 \
    --allow-unauthenticated
"""

import os, time, base64, io, json
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

# Global model state (loaded on first request)
_model = None
_task = None
_image_history = []
_stats = None


def _load_model():
    global _model, _stats
    import jax
    if not hasattr(jax, 'tree_map'):
        jax.tree_map = jax.tree_util.tree_map
    from octo.model.octo_model import OctoModel

    _model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")
    _stats = _model.dataset_statistics["bridge_dataset"]["action"]
    print(f"[CloudRun] Octo loaded on {jax.devices()}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "octo-small-1.5",
        "gpu": "L4",
        "loaded": _model is not None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    global _model, _task, _image_history, _stats
    import jax

    if _model is None:
        _load_model()

    data = request.get_json()

    if data.get("reset", False):
        _image_history = []
        _task = None
        return jsonify({"status": "ok"})

    lang = data.get("language", "pick up the object")
    if _task is None:
        _task = _model.create_tasks(texts=[lang])

    img_bytes = base64.b64decode(data["image_b64"])
    img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    if img.shape[:2] != (256, 256):
        img = np.array(Image.fromarray(img).resize((256, 256)))

    _image_history.append(img)
    if len(_image_history) > 2:
        _image_history = _image_history[-2:]

    history = list(_image_history)
    while len(history) < 2:
        history.insert(0, history[0])

    images = np.stack(history)[np.newaxis]
    pad_mask = np.zeros((1, 2), dtype=bool)
    pad_mask[0, -len(_image_history):] = True

    t0 = time.time()
    actions = _model.sample_actions(
        {"image_primary": images, "pad_mask": pad_mask},
        _task,
        unnormalization_statistics=_stats,
        rng=jax.random.PRNGKey(int(t0 * 1000) & 0xFFFFFFFF),
    )
    inference_ms = (time.time() - t0) * 1000

    a = np.array(actions)
    if a.ndim == 2:
        a = a[np.newaxis]

    return jsonify({
        "status": "ok",
        "action": {
            "end_effector_position": a[:, :, :3].tolist(),
            "end_effector_rotation": a[:, :, 3:6].tolist(),
            "gripper_close": a[:, :, 6:7].tolist(),
        },
        "inference_ms": round(inference_ms, 2),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
```

### 6.4 Dockerfile (for GCP Cloud Run)

```dockerfile
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY cloudrun_app.py /app/cloudrun_app.py
WORKDIR /app

CMD ["python3", "cloudrun_app.py"]
```

### 6.5 Server Requirements — `cloud_vla_server/requirements.txt`

```
# JAX with CUDA 12 support
jax[cuda12_pip]
# Octo VLA model and its data-loading dependency
git+https://github.com/kvablack/dlimp.git
git+https://github.com/octo-models/octo.git
# Image processing
Pillow
numpy
# Web server (Cloud Run only; Modal uses built-in web endpoints)
flask
gunicorn
```

---

## 7. Latency Optimization Strategies

### 7.1 Connection Pooling

```python
# Reuse TCP connections across inference calls (eliminates TLS handshake overhead)
import urllib3
self._http = urllib3.PoolManager(
    num_pools=2,
    maxsize=2,
    retries=urllib3.Retry(total=2, backoff_factor=0.1),
)
```

### 7.2 JPEG Compression

```python
# ~15 KB at quality=85 vs ~190 KB raw (12x smaller payload)
buf = io.BytesIO()
Image.fromarray(rgb).save(buf, format="JPEG", quality=85)
image_b64 = base64.b64encode(buf.getvalue()).decode()
```

### 7.3 Action Chunking (Amortized Latency)

```python
# Octo outputs H=4 actions per inference call
# At 60 Hz physics, inference runs every 4 steps = every 67 ms
# Network latency is amortized: even 150 ms round-trip is acceptable
# because the robot executes cached actions while waiting
```

### 7.4 Prefetch (Advanced)

```python
# Request next action chunk while executing current one
# Overlaps network I/O with physics simulation
# Requires threading (Isaac Sim's physics callback is synchronous)
import threading

def _prefetch_action(self):
    """Run in background thread during action execution."""
    self._next_action = self._client.get_action(self._latest_obs)

# In update_scenario():
if step_in_chunk == EXEC_STEPS - 2:  # One step before needing new action
    threading.Thread(target=self._prefetch_action, daemon=True).start()
```

---

## 8. Supported VLA Models

| Model | Params | HuggingFace ID | Inference (T4) | Min GPU | Notes |
|-------|--------|----------------|----------------|---------|-------|
| **Octo-Small** | 27M | `rail-berkeley/octo-small-1.5` | ~60 ms | T4 (2 GB) | Default, cheapest |
| **Octo-Base** | 93M | `rail-berkeley/octo-base-1.5` | ~80 ms | T4 (3 GB) | Better accuracy |
| **SmolVLA** | 450M | `lerobot/smolvla_base` | ~80 ms | T4 (4 GB) | Best generalization/size ratio |
| **OpenVLA-7B** | 7B | `openvla/openvla-7b` | ~200 ms | A10G (16 GB) | Best generalization |
| **OpenVLA-OFT** | 7B | `openvla/openvla-oft` | ~50 ms | A10G (16 GB) | 26x faster than base |
| **Pi0** | Large | `lerobot/pi0_base` | varies | A10G+ | Physical Intelligence |
| **GR00T N1.6** | 3B | `nvidia/GR00T-N1.6-3B` | 500–2000 ms | A100 (24 GB) | NVIDIA ecosystem |

To switch models on the server side, change the model ID in `load_model()` and
ensure the GPU has sufficient VRAM.

---

## 9. Cost Analysis

### Typical Research Usage: 4 hours/day, 22 days/month (88 hours)

| Platform | Billing Model | Gross Cost | Free Credits | **Net Monthly** |
|----------|--------------|-----------|-------------|-----------------|
| **Modal T4** | Per-second, scale-to-zero | $52 | $30/mo | **$22** |
| **HF Endpoint T4** | Per-minute, must stop manually | $44 | None | **$44** |
| **GCP Cloud Run L4** | Per-second, scale-to-zero | $59 | $300 trial | **$0** (first 5 mo) |
| **Colab Free** | Free (12h sessions) | $0 | Unlimited | **$0** |

### Heavy Usage: 8 hours/day, 22 days/month (176 hours)

| Platform | Net Monthly |
|----------|-------------|
| **Modal T4** | $74 |
| **HF Endpoint T4** | $88 |
| **GCP Cloud Run L4** | $118 |
| **GCP GKE Spot L4** | $33 (cheapest at scale, but complex) |

### GPU Pricing Reference (per hour, on-demand)

| GPU | Modal | HF Endpoints | GCP Cloud Run | GCP GKE |
|-----|-------|-------------|---------------|---------|
| T4 (16 GB) | $0.59 | $0.50 | N/A | $0.27 |
| L4 (24 GB) | N/A | $0.80 | $0.67 | $0.70 |
| A10G (24 GB) | $1.10 | $1.00 | N/A | N/A |
| A100 40 GB | $3.00 | $4.00 | N/A | $3.67 |

---

## 10. Implementation Order

### Phase 1: Modal Server (30 min)

1. `pip install modal && modal setup`
2. Create `cloud_vla_server/modal_app.py`
3. `modal deploy cloud_vla_server/modal_app.py`
4. Test with curl: `curl https://<url>/health`

### Phase 2: Isaac Sim Client (2–3 hours)

1. Copy `scenario_tashan_colab_vla.py` as starting point
2. Replace WebSocket client (`websockets` library) with HTTP client (`urllib3`)
3. Add backend selection logic (Modal / HF / GCP / Colab)
4. Keep all other code identical:
   - Sensor integration (`_integrate_sensor`, `_fix_rigid_body_transforms`, etc.)
   - Jacobian IK pipeline
   - Camera setup (Bridge V2 shoulder view)
   - Cube/target creation
   - Data logging and plotting
5. Update `ui_builder.py` import

### Phase 3: Test End-to-End (1 hour)

1. Load scenario in Isaac Sim
2. Set Modal endpoint URL
3. Run pick-and-place with "pick up the blue cube"
4. Verify latency: should see 80–150 ms round-trip
5. Check plots: actions, latency breakdown, tactile data

### Phase 4: Additional Backends (optional, 1–2 hours each)

1. HuggingFace Inference Endpoint — create model repo + deploy
2. GCP Cloud Run — build Docker image + `gcloud run deploy`
3. Test fallback chain: Modal → HF → GCP → Colab → Scripted

---

## 11. Comparison with Existing Colab Approach

| Aspect | Colab (existing) | Cloud VLA (new) |
|--------|-----------------|-----------------|
| Protocol | WebSocket + ngrok | REST HTTPS |
| URL management | Manual copy-paste per session | Fixed endpoint URL |
| Session lifetime | 12h free / 24h Pro | Unlimited |
| Reconnection | Manual | Automatic (HTTP is stateless) |
| Server setup | Run notebook cells manually | One-time `modal deploy` |
| Multi-backend | Single backend | 4 backends with auto-fallback |
| Cost | $0 (free tier) | $0–74/mo depending on usage |
| Latency | 200–500 ms (ngrok overhead) | 80–150 ms (direct HTTPS) |
| Reliability | Poor (session drops) | Excellent (managed infra) |

The existing `scenario_tashan_colab_vla.py` and `colab_vla_server.ipynb` remain
available as the free-tier fallback option within the new unified scenario.

---

## 12. References

- [Modal.com Pricing](https://modal.com/pricing) — GPU rates, free tier
- [Modal GPU Docs](https://modal.com/docs/reference/modal.gpu) — T4/A10G/A100/H100
- [HuggingFace Inference Endpoints](https://huggingface.co/docs/inference-endpoints) — Deployment guide
- [HF Endpoints Pricing](https://huggingface.co/docs/inference-endpoints/en/pricing) — GPU rates
- [GCP Cloud Run GPUs](https://docs.cloud.google.com/run/docs/configuring/services/gpu) — L4 GPU support
- [GCP Cloud Run Pricing](https://cloud.google.com/run/pricing) — Per-second billing
- [Octo Project](https://octo-models.github.io/) — Model architecture, training data
- [Octo HuggingFace](https://huggingface.co/rail-berkeley/octo-base) — Model weights
- [SmolVLA Blog](https://huggingface.co/blog/smolvla) — 450M VLA, 487 datasets
- [OpenVLA](https://openvla.github.io/) — 7B VLA, Bridge V2 trained
- [OpenVLA-OFT](https://openvla-oft.github.io/) — 26x faster parallel decoding
- [LeRobot v0.4.0](https://huggingface.co/blog/lerobot-release-v040) — Pi0, SmolVLA
