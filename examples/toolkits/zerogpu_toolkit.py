# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========

import os

from camel.agents import ChatAgent
from camel.toolkits import FunctionTool, ZeroGPUToolkit

"""
Example usage of ZeroGPUToolkit.

Before running, set your environment variables:

export ZEROGPU_API_KEY="your-api-key"
export ZEROGPU_PROJECT_ID="your-project-id"
"""

toolkit = ZeroGPUToolkit(
    api_key=os.getenv("ZEROGPU_API_KEY"),
    project_id=os.getenv("ZEROGPU_PROJECT_ID"),
)


# =========================
# Direct Toolkit Usage
# =========================

print("=== Summarization ===")
print(toolkit.zerogpu_summarize("AI is transforming industries rapidly."))
"""
===============================================================================
AI is rapidly transforming industries by improving efficiency, enabling
automation, and unlocking new data-driven insights across sectors.
===============================================================================
"""

print("\n=== Classification ===")
print(toolkit.zerogpu_classify_iab("Latest football match results"))
"""
===============================================================================
{"category": "sports"}
===============================================================================
"""

print("\n=== PII Redaction ===")
print(toolkit.zerogpu_redact_pii("Contact John at john@example.com"))
"""
===============================================================================
Contact [PERSON] at [EMAIL]
===============================================================================
"""


# =========================
# Agent Integration
# =========================

agent = ChatAgent(
    system_message="You are a helpful assistant.",
    tools=[
        FunctionTool(toolkit.zerogpu_summarize),
        FunctionTool(toolkit.zerogpu_classify_iab),
        FunctionTool(toolkit.zerogpu_redact_pii),
    ],
)

response = agent.step(
    input_message="Summarize this: AI is revolutionizing healthcare.",
    response_format=None,
)

print("\n=== Agent Response ===")
print(response.msgs[0].content)
"""
===============================================================================
AI is revolutionizing healthcare by enabling faster diagnoses, personalized
treatments, and improved patient outcomes through advanced data analysis.
===============================================================================
"""
