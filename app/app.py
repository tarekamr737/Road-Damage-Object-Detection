from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from PIL import Image

from road_damage.inference.annotation import (
    CLASS_COLORS,
    annotate_image,
    detection_rows,
    detections_to_csv,
    detections_to_json,
    filter_detections,
    image_to_png_bytes,
)
from road_damage.inference.input_validation import read_uploaded_image
from road_damage.inference.roboflow_service import (
    RoboflowHostedModel,
    load_roboflow_model,
    predict_roboflow_image,
)
from road_damage.inference.service import load_model, predict_image
from road_damage.inference.startup import acceleration_notice, deployment_health_errors
from road_damage.training.runtime import load_config

APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = APP_ROOT / "configs" / "deployment" / "app.yaml"


st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="road",
    layout="wide",
    initial_sidebar_state="auto",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --rd-bg: #0b0d0c;
          --rd-surface: #111512;
          --rd-surface-2: #171c18;
          --rd-elevated: #1d231f;
          --rd-ink: #eef3ee;
          --rd-muted: #a8b2aa;
          --rd-soft: #757f77;
          --rd-charcoal: #0f1210;
          --rd-amber: #f1c453;
          --rd-amber-strong: #ffd56a;
          --rd-green: #45d483;
          --rd-red: #ff6b5f;
          --rd-blue: #7db7ff;
          --rd-border: #2f3832;
          --rd-border-strong: #48534b;
          --rd-shadow: rgba(0, 0, 0, .35);
        }
        .stApp {
          background: var(--rd-bg);
          color: var(--rd-ink);
          overflow-x: hidden;
        }
        html,
        body,
        [class*="css"] {
          font-family: "Aptos", "Segoe UI", sans-serif;
        }
        .block-container {
          padding-top: 2rem;
          padding-bottom: 3rem;
        }
        h1, h2, h3 {
          letter-spacing: 0;
          color: var(--rd-ink);
          font-family: "Bahnschrift", "Segoe UI Semibold", sans-serif;
          font-weight: 700;
        }
        .rd-topline {
          position: relative;
          border-top: 1px solid var(--rd-border);
          border-bottom: 1px solid var(--rd-border);
          padding: 18px 0 16px;
          margin-bottom: 12px;
        }
        .rd-topline::before {
          background: var(--rd-amber);
          content: "";
          height: 3px;
          left: 0;
          position: absolute;
          top: -1px;
          width: 112px;
        }
        .rd-topline h1 {
          color: var(--rd-ink);
          font-size: 3rem;
          line-height: 1;
          margin: 0 0 10px;
        }
        .rd-kicker {
          color: var(--rd-amber);
          text-transform: uppercase;
          font-size: .74rem;
          font-weight: 800;
          letter-spacing: .08rem;
          margin-bottom: 8px;
        }
        .rd-sub {
          color: var(--rd-muted);
          max-width: 860px;
          overflow-wrap: anywhere;
          line-height: 1.55;
        }
        .rd-command-strip {
          background: var(--rd-surface);
          border: 1px solid var(--rd-border);
          border-radius: 8px;
          box-shadow: 0 16px 40px var(--rd-shadow);
          display: flex;
          flex-wrap: wrap;
          gap: 0;
          margin: 0 0 12px;
          overflow: hidden;
        }
        .rd-command-item {
          border-right: 1px solid var(--rd-border);
          flex: 1 1 280px;
          min-width: 0;
          padding: 12px 14px;
        }
        .rd-command-item:last-child {
          border-right: 0;
        }
        .rd-command-item span {
          color: var(--rd-soft);
          display: block;
          font-size: .74rem;
          font-weight: 800;
          line-height: 1.2;
          margin-bottom: 4px;
          text-transform: uppercase;
        }
        .rd-command-item strong {
          color: var(--rd-ink);
          display: block;
          font-size: .96rem;
          line-height: 1.35;
          overflow-wrap: anywhere;
        }
        .rd-status-grid {
          display: grid;
          gap: 10px;
          grid-template-columns: repeat(2, minmax(0, 1fr));
          margin-bottom: 18px;
        }
        .rd-status {
          border: 1px solid var(--rd-border);
          border-left: 4px solid var(--rd-green);
          border-radius: 8px;
          padding: 12px 14px;
          background: var(--rd-surface-2);
          color: var(--rd-muted);
          overflow-wrap: anywhere;
          box-shadow: 0 12px 30px var(--rd-shadow);
        }
        .rd-status strong {
          color: var(--rd-ink);
        }
        .rd-warning {
          border-left-color: var(--rd-red);
        }
        .rd-note {
          border-left-color: var(--rd-amber);
        }
        div[data-testid="stMetric"] {
          background: var(--rd-surface-2);
          border: 1px solid var(--rd-border);
          border-left: 4px solid var(--rd-amber);
          border-radius: 8px;
          padding: 12px 14px;
          box-shadow: 0 12px 30px var(--rd-shadow);
        }
        div[data-testid="stMetricLabel"] p {
          color: var(--rd-muted);
          font-weight: 700;
        }
        div[data-testid="stMetricValue"] {
          color: var(--rd-ink);
          font-family: "Bahnschrift", "Segoe UI Semibold", sans-serif;
        }
        div[data-testid="stImage"] img {
          border: 1px solid var(--rd-border);
          border-radius: 8px;
          box-shadow: 0 18px 48px var(--rd-shadow);
        }
        .stButton > button {
          border-radius: 8px;
          border: 1px solid var(--rd-amber-strong);
          background: var(--rd-amber);
          color: #15110a;
          font-weight: 800;
          letter-spacing: 0;
          min-height: 2.8rem;
        }
        .stButton > button:hover {
          border-color: #ffe08a;
          background: var(--rd-amber-strong);
          color: #15110a;
        }
        .stDownloadButton > button {
          border-radius: 8px;
          border: 1px solid var(--rd-border-strong);
          background: var(--rd-elevated);
          color: var(--rd-ink);
          font-weight: 700;
        }
        .stDownloadButton > button:hover {
          border-color: var(--rd-amber);
          color: var(--rd-amber-strong);
        }
        section[data-testid="stSidebar"] {
          background: var(--rd-surface);
          border-right: 1px solid var(--rd-border);
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {
          color: var(--rd-ink);
        }
        section[data-testid="stSidebar"] small,
        [data-testid="stCaptionContainer"] {
          color: var(--rd-muted);
        }
        [data-testid="stFileUploaderDropzone"] {
          background: var(--rd-surface-2);
          border: 1px dashed var(--rd-border-strong);
          border-radius: 8px;
        }
        [data-testid="stFileUploaderDropzone"]:hover {
          border-color: var(--rd-amber);
        }
        button[data-baseweb="tab"] {
          color: var(--rd-muted);
          font-weight: 800;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
          color: var(--rd-amber-strong);
        }
        div[data-baseweb="select"] > div,
        input,
        textarea {
          background: var(--rd-elevated);
          border-color: var(--rd-border);
          color: var(--rd-ink);
        }
        code {
          background: var(--rd-charcoal);
          border: 1px solid var(--rd-border);
          border-radius: 6px;
          color: var(--rd-amber-strong);
          padding: 2px 5px;
        }
        @media (max-width: 700px) {
          .block-container,
          div[data-testid="stMainBlockContainer"] {
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100vw;
            min-width: 0;
          }
          .rd-topline {
            box-sizing: border-box;
            padding-top: 16px;
            width: 100%;
          }
          .rd-topline h1 {
            font-size: 2.05rem;
            line-height: 1.05;
            white-space: normal;
            overflow-wrap: anywhere;
          }
          .rd-command-strip {
            display: block;
          }
          .rd-command-item {
            border-bottom: 1px solid var(--rd-border);
            border-right: 0;
          }
          .rd-command-item:last-child {
            border-bottom: 0;
          }
          .rd-status-grid {
            grid-template-columns: 1fr;
          }
          .rd-status {
            box-sizing: border-box;
            width: 100%;
          }
          .rd-sub,
          .rd-status,
          .rd-command-item,
          .rd-status * {
            white-space: normal;
            word-break: break-word;
            overflow-wrap: anywhere;
          }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def app_config() -> dict:
    return load_config(CONFIG_PATH)


@st.cache_resource
def load_selected_model(
    provider: str,
    model_path: str,
    device: str,
    workspace: str,
    project: str,
    version: int,
    endpoint: str,
) -> Any | RoboflowHostedModel:
    if provider == "local_ultralytics":
        return load_model(APP_ROOT / model_path, device=device)
    if provider != "roboflow_hosted":
        raise RuntimeError(f"Unsupported deployment provider: {provider}")
    return load_roboflow_model(
        workspace,
        project,
        version,
        env_path=APP_ROOT / ".env",
        endpoint=endpoint,
    )


def runtime_label(config: dict) -> str:
    provider = str(config.get("provider", "roboflow_hosted"))
    if provider == "local_ultralytics":
        return f"local Ultralytics checkpoint - {config['model_path']}"
    if provider == "roboflow_hosted":
        return f"hosted Roboflow inference - version {config['roboflow_version']}"
    return provider


def read_image(
    uploaded_file,
    max_upload_mb: int,
    max_dimension: int,
    supported_types: list[str],
) -> Image.Image:
    return read_uploaded_image(
        uploaded_file,
        max_upload_mb=max_upload_mb,
        max_dimension=max_dimension,
        supported_types=supported_types,
    )


def verify_startup(config: dict) -> None:
    errors = deployment_health_errors(config, APP_ROOT)
    if not errors:
        return

    st.error("Deployment startup check failed.")
    for error in errors:
        st.error(error)
    st.stop()


def render_header(config: dict) -> None:
    model_name = config.get("model_name") or config.get("roboflow_model_id")
    st.markdown(
        """
        <div class="rd-topline">
          <div class="rd-kicker">Production inspection console</div>
          <h1>Road Damage Detection</h1>
          <div class="rd-sub">
            Localized pavement damage review for field images, QA samples,
            and deployment smoke checks.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="rd-command-strip">
          <div class="rd-command-item">
            <span>Active model</span>
            <strong>{model_name}</strong>
          </div>
          <div class="rd-command-item">
            <span>Runtime</span>
            <strong>{runtime_label(config)}</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if str(config.get("provider")) == "local_ultralytics":
        acceleration_message, acceleration_warning = acceleration_notice(config)
        status_class = "rd-status rd-warning" if acceleration_warning else "rd-status"
        st.markdown(
            f"""
            <div class="rd-status-grid">
              <div class="rd-status">
                <strong>Local production runtime</strong><br>
                Images stay on this machine. Local YOLOv8s is the active
                production fallback.
              </div>
              <div class="{status_class}">
                <strong>Acceleration</strong><br>
                {acceleration_message}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="rd-status-grid">
          <div class="rd-status rd-warning">
            <strong>Hosted inference</strong><br>
            Submitted images are sent to Roboflow.
          </div>
          <div class="rd-status rd-note">
            <strong>Review required</strong><br>
            Predictions need human validation before repair or safety decisions.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls(config: dict) -> tuple[float, float, list[str], int]:
    st.sidebar.header("Detection")
    confidence = st.sidebar.slider(
        "Confidence",
        min_value=0.05,
        max_value=0.95,
        value=float(config["default_confidence_threshold"]),
        step=0.05,
        help=(
            "Lower values show more candidates; higher values keep stronger detections."
        ),
    )
    overlap = st.sidebar.slider(
        "Overlap",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Higher values allow overlapping boxes to remain separate.",
    )
    classes = st.sidebar.multiselect(
        "Classes",
        options=list(CLASS_COLORS),
        default=list(CLASS_COLORS),
        help="Limit the output to selected damage types.",
    )
    max_detections = st.sidebar.number_input(
        "Maximum detections",
        min_value=1,
        max_value=300,
        value=100,
        step=1,
        help="Cap the displayed detections after confidence sorting.",
    )
    if str(config.get("provider")) == "roboflow_hosted":
        st.sidebar.caption(
            "Hosted model access requires `ROBOFLOW_API_KEY` in the local environment."
        )
    else:
        st.sidebar.caption("Local checkpoint inference is active.")
    return confidence, overlap, classes, int(max_detections)


def predict_selected_model(
    provider: str,
    model: Any | RoboflowHostedModel,
    image: Image.Image,
    confidence: float,
    overlap: float,
    max_detections: int,
    config: dict,
):
    if provider == "local_ultralytics":
        return predict_image(
            model,
            image,
            confidence=confidence,
            iou=overlap,
            image_size=int(config["default_image_size"]),
            device=None,
            max_det=max_detections,
        )
    return predict_roboflow_image(
        model,
        image,
        confidence=confidence,
        overlap=overlap,
    )


def render_results(
    image: Image.Image,
    annotated: Image.Image,
    result,
    filtered,
    confidence: float,
    overlap: float,
) -> None:
    total = len(filtered)
    highest = max((detection.confidence for detection in filtered), default=0.0)
    fps = 1000 / result.inference_ms if result.inference_ms > 0 else 0

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Detections", total)
    col_b.metric("Highest Confidence", f"{highest:.0%}")
    col_c.metric("Request Time", f"{result.inference_ms:.0f} ms")
    col_d.metric("Approx. FPS", f"{fps:.2f}")

    original_col, annotated_col = st.columns(2)
    original_col.image(image, caption="Original", use_container_width=True)
    annotated_col.image(annotated, caption="Annotated", use_container_width=True)

    rows = detection_rows(filtered)
    if rows:
        count_rows = (
            pd.DataFrame(rows)["class_name"]
            .value_counts()
            .rename_axis("damage_type")
            .reset_index(name="count")
        )
        st.dataframe(count_rows, use_container_width=True, hide_index=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            (
                '<div class="rd-status rd-warning">'
                "No detections matched the active filters.</div>"
            ),
            unsafe_allow_html=True,
        )

    settings = {
        "confidence": confidence,
        "overlap": overlap,
        "detections": total,
    }
    st.caption("Active settings: " + json.dumps(settings))

    left, right = st.columns(2)
    left.download_button(
        "Download annotated image",
        data=image_to_png_bytes(annotated),
        file_name="road_damage_annotated.png",
        mime="image/png",
        use_container_width=True,
    )
    right.download_button(
        "Download detections CSV",
        data=detections_to_csv(filtered),
        file_name="road_damage_detections.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.download_button(
        "Download detections JSON",
        data=detections_to_json(filtered),
        file_name="road_damage_detections.json",
        mime="application/json",
        use_container_width=True,
    )


def main() -> None:
    inject_css()
    config = app_config()
    verify_startup(config)
    render_header(config)
    confidence, overlap, classes, max_detections = sidebar_controls(config)

    input_tab, camera_tab = st.tabs(["Upload", "Camera"])
    uploaded = input_tab.file_uploader(
        "Image file",
        type=config["supported_types"],
        label_visibility="collapsed",
    )
    captured = camera_tab.camera_input("Camera image", label_visibility="collapsed")
    selected_file = uploaded or captured

    if selected_file is None:
        st.markdown(
            '<div class="rd-status rd-warning">Waiting for an image.</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        image = read_image(
            selected_file,
            max_upload_mb=int(config["max_upload_mb"]),
            max_dimension=int(config["max_image_dimension"]),
            supported_types=list(config["supported_types"]),
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    st.image(image, caption="Ready for detection", use_container_width=True)
    run = st.button("Run Detection", type="primary", use_container_width=True)
    if not run:
        return

    try:
        provider = str(config.get("provider", "roboflow_hosted"))
        model = load_selected_model(
            provider,
            str(config["model_path"]),
            str(config.get("device", "auto")),
            str(config["roboflow_workspace"]),
            str(config["roboflow_project"]),
            int(config["roboflow_version"]),
            str(config["roboflow_endpoint"]),
        )
        with st.spinner("Running inference..."):
            result = predict_selected_model(
                provider,
                model,
                image,
                confidence,
                overlap,
                max_detections,
                config,
            )
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        return

    filtered = filter_detections(
        result.detections,
        min_confidence=confidence,
        classes=set(classes) if classes else set(),
        max_detections=max_detections,
    )
    annotated = annotate_image(image, filtered)
    render_results(image, annotated, result, filtered, confidence, overlap)


if __name__ == "__main__":
    main()
