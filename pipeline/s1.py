import json
from pathlib import Path

# Load OpenAPI spec (replace with your file path if needed)
openapi_file = Path("docs.json")

# Explicitly read as UTF-8 to avoid UnicodeDecodeError
spec = json.loads(openapi_file.read_text(encoding="utf-8"))

# Extract only /predict endpoint
predict_info = {}

if "/predict" in spec.get("paths", {}):
    methods = spec["paths"]["/predict"]
    for method, details in methods.items():
        predict_info = {
            "path": "/predict",
            "method": method.upper(),
            "operationId": details.get("operationId"),
            "summary": details.get("description", "").strip(),
            "tags": details.get("tags", []),
            "requestBody": details.get("requestBody", {}),
            "responses": list(details.get("responses", {}).keys()),
        }

# Structure everything neatly for an AI agent
parsed_spec = {
    "predict_endpoint": predict_info
}

# Show result (could also dump to JSON file)
api_info = json.dumps(parsed_spec, indent=2, ensure_ascii=False)
print(api_info)