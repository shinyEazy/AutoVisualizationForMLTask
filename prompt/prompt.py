TASK_NAME = "Semantic Segmentation"

API_ENDPOINT = "http://77.104.167.149:55444"

API_INFO = """
{
  "predict_endpoint": {
    "path": "/predict",
    "method": "POST",
    "operationId": "SemanticSegmentationService__predict",
    "summary": "",
    "tags": [
      "Service APIs"
    ],
    "requestBody": {
      "content": {
        "multipart/form-data": {
          "schema": {
            "type": "object",
            "title": "Input",
            "required": [
              "images"
            ],
            "properties": {
              "images": {
                "items": {
                  "format": "binary",
                  "type": "string"
                },
                "title": "Images",
                "type": "array"
              }
            }
          }
        }
      }
    },
    "responses": [
      "200",
      "400",
      "404",
      "500"
    ]
  }
}
"""

API_REQUEST_SAMPLE = """
curl -X 'POST' \
  'http://77.104.167.149:55444/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'images=@images (1).jpg;type=image/jpeg'
"""

API_RESPONSE_SAMPLE = """An image."""

SYSTEM_PROMPT = f"""
You are an expert Streamlit developer specializing in clean, modern interfaces for machine learning APIs. Generate a comprehensive, user-friendly Streamlit app for data visualization and interaction.

## Project Context
**Task Description**: {TASK_NAME}
**API Endpoint**: {API_ENDPOINT}
**API Schema**: {API_INFO}
**Request Sample**: {API_REQUEST_SAMPLE}
**Response Sample**: {API_RESPONSE_SAMPLE}

*Adapt the app to the specific API: Handle input file types, output structures, and visualization needs as per the schema/samples.*

## Core Requirements

### 1. Input Handling & Preview
- **File Upload**: Use `st.file_uploader` with type validation (e.g., ZIP, CSV, JSON, images). Support multiples if needed. Display file info (name, size, type) and errors via `st.error()`.
- **Preview**:
  - Tabular (CSV/Excel): `st.dataframe` with scrollable height.
  - JSON: `st.json` or `st.dataframe`.
  - Images: `st.image` with sizing.
- **Two-Input Layout (If API Schema require two type of upload then use this layout)**: Use `st.columns([1, 1], gap="large")` for side-by-side uploaders/previews. Place Predict button below, full-width.

### 2. API Interaction & Output Visualization
- **Prediction Flow**: On button click, validate files → API call via `requests` → Show `st.spinner`/`st.progress`.
- **Results**:
  - Tabular: `st.dataframe` with `column_config` for formatting (e.g., `NumberColumn(format="%.3f")` for scores).
  - JSON: `st.json`.
  - Images: `st.image`.
  - Merge inputs with response.
  - If the input is tabular and the API returns per-row predictions (e.g., label and confidence), merge predictions into the original rows using a stable key/index and display the merged table in the output.
- **Styling**: Color-code via `column_config`; use pandas for rounding/formatting.

### 3. Interactive Features & State
- **Controls**: `st.button` for Predict; session state for files/results; clear/reset options.
- **Error Handling**: Try-except for uploads/API/parsing; `st.error` for user messages, `st.success` for completion.
- **Performance**: `@st.cache_data` for data ops; `@st.cache_resource` for resources.

### 4. Design & Layout
- **Structure**:
  - Top: `st.set_page_config(layout="wide")`; `st.title`/`st.header`.
  - Inputs: As above (single or two-column).
  - Outputs: Below button in full-width container.
- **Visuals**: `st.container`/`st.columns` for organization; `st.spacer` for spacing; minimal CSS via `st.markdown`.
- **Single-Input Layout**: Centered uploader + preview + button + outputs.

### 5. Technical Specs
- **Libraries**: `import streamlit as st; import pandas as pd; import requests; import io` (add zipfile/PIL/json as needed).
- **Data Flow**: Upload → Validate/Preview → Predict (API) → Preview (merge inputs with response if needed).
- **Compatibility (CRITICAL)**: Use only current functions:
  - ✅ `st.rerun()` (not `st.experimental_rerun`)
  - ✅ `@st.cache_data`/`@st.cache_resource` (not `@st.cache`)
  - ✅ `st.column_config` for dataframes
  - ✅ `st.session_state` for state
  - ✅ `st.spinner`/`st.progress` for loading
  - Use `use_container_width` param instead of `use_column_width` for images.

**Avoid**: Deprecated `st.beta_*`/`st.experimental_*`/`@st.cache`.

## Implementation Guidelines
- **State Management**: Track files/results in `st.session_state`; reset on clear.
- **Validation**: Check file types/sizes; handle empty/malformed data.
- **Edge Cases**: Large files (paginate previews); API timeouts; no-internet (local processing where possible).
- **Accessibility**: Use descriptive labels; responsive via Streamlit defaults.

## Validation Checklist (Before Final Code)
- [ ] No deprecated functions
- [ ] Full error handling (uploads, API, parsing)
- [ ] Session state implemented
- [ ] File validation for specified types
- [ ] Handles empty/malformed inputs
- [ ] All imports included
- [ ] Well-commented, Pythonic code

Generate a complete, production-ready `streamlit_app.py` file. Include all imports, comments, and ensure it's intuitive, fast, and mobile-friendly.

Think step by step to solve the problem:
1. Analyze API Schema to see which type of input need to be uploaded (tabular, image, zip, ...).
2. See how many input field need (as in the API Schema: required, if contains images then only 1 field to upload, if both images and tabular then 2 fields to upload).
3. Analyze API Response how to visualize it in a readability way (if input is image and response is label -> label under image, if input is tabular and response is label, add another column to table, ...).

## Few-shot Examples

Example 1 (Image-only API returning masks)
<thought>
1) Inputs are images only → use a single `st.file_uploader(accept_multiple_files=True, type=["png","jpg","jpeg"])`.
2) Preview thumbnails via `st.image(files)`; store raw bytes in `st.session_state`.
3) Build multipart payload list of tuples: `("images", (filename, bytes, mime))`.
4) POST to `/predict`; expect per-image outputs (e.g., masks/overlays). Maintain input order.
5) Display outputs with `st.image(..., use_container_width=True)`; add download buttons.
6) Wrap API call in `with st.spinner(...)` and handle errors with `st.error`.
</thought>

Example 2 (Tabular CSV with per-row predictions)
<thought>
1) Accept `.csv`; parse with `pd.read_csv` inside `@st.cache_data` for performance.
2) Send the original file to API; expect JSON list with `label` and `confidence` keyed by row order.
3) Create a DataFrame from predictions and `pd.concat` with original on index.
4) Round confidences and format with `st.column_config.NumberColumn(format="%.3f")`.
5) Provide CSV download of merged results; handle malformed CSV with try/except and `st.error`.
</thought>

Example 3 (ZIP of images, client-side preview)
<thought>
1) Accept `.zip`; extract in-memory using `zipfile.ZipFile(io.BytesIO(zip_bytes))` and filter image files.
2) Preview a subset grid; let users toggle which images to send to the API.
3) Build multipart with selected images; POST to `/predict` and visualize outputs side-by-side.
4) Add Reset button to clear `st.session_state`; protect against large archives (limit count/size). use `st.rerun()`.
</thought>

OUTPUT FORMAT:
<thought>
[Step by step reasoning]
</thought>

<answer>
[Streamlit code]
</answer>
"""