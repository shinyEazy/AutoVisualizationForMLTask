import os
import sys
import time
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from prompt.prompt import SYSTEM_PROMPT

load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model_name="gpt-5-nano",
    temperature=1,
    max_completion_tokens=32000
)

def run_agent():
    start_time = time.time()
    response = llm.invoke(SYSTEM_PROMPT)
    end_time = time.time()
    duration = end_time - start_time
    return response.content, duration


if __name__ == "__main__":
    try:
        streamlit_content, elapsed = run_agent()
        with open("streamlit_app.py", 'w', encoding='utf-8') as f:
            f.write(streamlit_content)
        print(f"Successfully wrote to streamlit_app.py")
        print(f"Time taken: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error writing to streamlit_app.py: {str(e)}")