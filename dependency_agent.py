import os
import sys
from google import genai

from aura.agent_logic import DependencyAgent
from aura.dependency_agent_template import GeminiClientWrapper


AGENT_CONFIG = {
    "PROJECT_NAME": "attention-is-all-you-need-pytorch_aura",
    "IS_INSTALLABLE_PACKAGE": False,
    "REQUIREMENTS_FILE": "requirements.txt",
    "METRICS_OUTPUT_FILE": "metrics_output.txt",
    "PRIMARY_REQUIREMENTS_FILE": "primary_requirements.txt",
    "VALIDATION_CONFIG": {
        "type": "script",
        "smoke_test_script": "valudation_aura.py",
        "project_dir": "."
    },
    "MAX_RUN_PASSES": 3
}


if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        sys.exit("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")

    llm_client = GeminiClientWrapper(
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.5-flash",
    )

    agent = DependencyAgent(config=AGENT_CONFIG, llm_client=llm_client)
    agent.run()
