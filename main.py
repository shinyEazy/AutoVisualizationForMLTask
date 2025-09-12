import subprocess
import sys
import time

if __name__ == "__main__":
    # Start server.py in the background
    server_process = subprocess.Popen([sys.executable, "server.py"])
    
    try:
        # Wait a bit to ensure server is up
        time.sleep(2)
        
        # Run pipeline.py with correct API endpoint
        subprocess.run([
            sys.executable, "pipeline.py",
            "--task_name", "MULTIMODAL_CLASSIFICATION22",
            "--input_type", "image, text",
            "--api_endpoint", "http://127.0.0.1:8000/predict"
        ], check=True)
    
    finally:
        # Clean up server process when done
        server_process.terminate()
        server_process.wait()
