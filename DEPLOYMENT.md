# Streamlit Deployment

## Chosen Platform

The project deployment target is a self-hosted Streamlit application using the
local YOLOv8s PyTorch checkpoint. The app runs at:

```text
http://localhost:8501
```

This platform was selected because the production model is local, Roboflow
hosted inference is currently credit-blocked, and the app does not require
external secrets for the active `local_ultralytics` runtime.

## Required Artifact

The Docker/self-hosted deployment expects the production checkpoint at:

```text
models/exports/production_road_damage_model.pt
```

Expected SHA256:

```text
9c51b412886730363d100577098a04fc92c7e5a6c0481b786a796ee3de9d5d13
```

The startup health check verifies this checksum before the app serves
predictions.

## Local Deployment

Run the deployment smoke test:

```powershell
python scripts/deployment_smoke.py
```

Start the app:

```powershell
streamlit run app/app.py --server.address 0.0.0.0 --server.port 8501
```

Open:

```text
http://localhost:8501
```

## Docker Deployment

Docker is optional, but the repository includes a Dockerfile and Compose file
for a repeatable self-hosted deployment.

```powershell
docker compose up --build
```

The image copies only the active production checkpoint, not `.env`, datasets,
training runs, notebooks, or evaluation caches.

## Last Local Verification

The current local deployment smoke artifacts are saved under
`reports/deployment/`:

- `deployment_smoke.json`
- `streamlit_launch_smoke.json`
- `app_home_desktop.png`
- `app_home_mobile.png`

The latest launch smoke reported health `200 ok` at `http://localhost:8501`.
The model smoke verified the checkpoint checksum, upload-size agreement, model
load, first synthetic inference, and process RSS under the configured 4096 MB
limit.

## Secrets

No production secret is required for the selected local YOLOv8s runtime. If the
app is later switched to `roboflow_hosted`, store `ROBOFLOW_API_KEY` as a
platform environment variable or secret. Never copy `.env` into the image.

## Limits

- Maximum upload size: 15 MB
- Maximum decoded image dimension: 4096 px
- Expected local URL: `http://localhost:8501`
- Active runtime: PyTorch `.pt`, not ONNX
- Memory smoke threshold: 4096 MB process RSS

## Licensing

The local model was trained from a Roboflow dataset whose local metadata records
`CC BY 4.0`. Keep source attribution with public demos, reports, and any
redistributed dataset-derived artifacts.
