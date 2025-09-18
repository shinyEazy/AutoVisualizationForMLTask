TASK_NAME = "Multimodal Classification"

API_ENDPOINT = "http://127.0.0.1:8000"

API_INFO = """
{
  "predict_endpoint": {
    "path": "/predict",
    "method": "POST",
    "operationId": "MultimodalClassificationService__predict",
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
              "image_zip",
              "labels_csv"
            ],
            "properties": {
              "image_zip": {
                "description": "A ZIP file containing all the images for multimodal classification task.",
                "format": "binary",
                "title": "Image Zip",
                "type": "string"
              },
              "labels_csv": {
                "description": "A csv file for multimodal classification task.",
                "format": "binary",
                "title": "Labels Csv",
                "type": "string"
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
  'http://47.186.63.142:52901/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image_zip=@super_small_image.zip;type=application/x-zip-compressed' \
  -F 'labels_csv=@super_small_data.csv;type=text/csv'
"""

API_RESPONSE_SAMPLE = """
{
    "status": "success",
    "infer_time": 0.5484125036746264,
    "predictions": [
        {
            "key": 0,
            "class": 0,
            "confidence": 0.94629967212677
        },
        {
            "key": 1,
            "class": 1,
            "confidence": 0.6092917323112488
        }
    ]
}
"""

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
- **Two-Input Layout**: Use `st.columns([1, 1], gap="large")` for side-by-side uploaders/previews. Place Predict button below, full-width.

### 2. API Interaction & Output Visualization
- **Prediction Flow**: On button click, validate files → API call via `requests` → Show `st.spinner`/`st.progress`.
- **Results**:
  - Tabular: `st.dataframe` with `column_config` for formatting (e.g., `NumberColumn(format="%.3f")` for scores).
  - JSON: `st.json`.
  - Images: `st.image`.
  - Merge inputs with response.
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
  - Sidebar: Optional controls/metrics via `st.sidebar`.
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
"""
