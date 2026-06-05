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
import json
import os
from typing import Any, Dict, List, Optional

from zerogpu import ZerogpuApi

from camel.logger import get_logger
from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.utils import (
    MCPServer,
    api_keys_required,
    dependencies_required,
)

logger = get_logger(__name__)


@MCPServer()
class ZeroGPUToolkit(BaseToolkit):
    r"""A toolkit for interacting with ZeroGPU-hosted models.

    This toolkit provides access to a variety of lightweight AI models
    deployed on the ZeroGPU platform for tasks such as:

    - Text generation
    - Translation
    - Summarization
    - Intent detection
    - PII extraction and redaction
    - IAB content classification (ad tech)
    - Email filtering

    Each method wraps a specific ZeroGPU model and exposes it as a tool
    compatible with CAMEL agents.
    """

    @dependencies_required("zerogpu")
    @api_keys_required(
        [
            ("api_key", "ZEROGPU_API_KEY"),
            ("project_id", "ZEROGPU_PROJECT_ID"),
        ]
    )
    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        r"""Initialize ZeroGPUToolkit.

        Args:
            api_key (Optional[str]): ZeroGPU API key.
                If not provided, will be read from ZEROGPU_API_KEY.
            project_id (Optional[str]): ZeroGPU project ID.
                If not provided, will be read from ZEROGPU_PROJECT_ID.
            base_url (Optional[str]): Base API URL.
                Defaults to environment or production endpoint.
            timeout (Optional[float]): Request timeout in seconds.
        """
        super().__init__(timeout=timeout)

        # Resolve credentials
        self.api_key = api_key or os.getenv("ZEROGPU_API_KEY")
        self.project_id = project_id or os.getenv("ZEROGPU_PROJECT_ID")

        # Validate API key
        if not self.api_key or not self.api_key.startswith("zgpu-api-"):
            raise ValueError(
                "Invalid ZeroGPU API key format (must start with 'zgpu-api-')"
            )

        # Resolve base URL
        self.base_url = (
            base_url
            or os.getenv("ZEROGPU_BASE_URL")
            or "https://api.zerogpu.ai"
        ).rstrip("/")

        self.client = ZerogpuApi(
            api_key=self.api_key,
            project_id=self.project_id,
            base_url=self.base_url,
        )

    def zerogpu_chat(self, text: str, system: Optional[str] = None) -> str:
        r"""Chat using LFM Instruct model.

        Use this for tasks that require natural language understanding,
        conversation, or dialog.

        Args:
            text (str): The input text.
            system (Optional[str], optional): The system prompt.
                Defaults to None.

        Returns:
            str: The response text.
        """

        input_data = [{"role": "user", "content": text}]
        if system:
            input_data.insert(0, {"role": "system", "content": system})

        response = self._run(
            model="LFM2.5-1.2B-Instruct",
            input_data=input_data,
        )

        if "error" in response:
            return f"Error: {response['error']}"

        return self._extract_text(response)

    def zerogpu_chat_thinking(
        self, text: str, system: Optional[str] = None
    ) -> str:
        r"""Generate responses using the LFM Thinking model.

        Use this for tasks requiring deeper reasoning, multi-step thinking,
        or more structured problem-solving compared to standard chat.

        Args:
            text (str): The user input message.
            system (Optional[str], optional): System prompt to guide behavior.

        Returns:
            str: The generated response.
        """

        input_data = [{"role": "user", "content": text}]

        if system:
            input_data.insert(0, {"role": "system", "content": system})

        response = self._run(
            model="LFM2.5-1.2B-Thinking",
            input_data=input_data,
        )

        if "error" in response:
            return f"Error: {response['error']}"

        return self._extract_text(response)

    def zerogpu_summarize(self, text: str) -> str:
        r"""Summarize input text into a concise form.

        Use this when you need to condense long content into key ideas
        while preserving the main meaning.

        Args:
            text (str): The text to summarize.

        Returns:
            str: A shortened version of the input text.
        """

        input_data = text

        response = self._run(
            model="llama-3.1-8b-instruct-fast",
            input_data=input_data,
        )

        if "error" in response:
            return f"Error: {response['error']}"

        return self._extract_text(response)

    def zerogpu_classify_iab(self, text: str) -> Dict[str, Any]:
        r"""Classify text into IAB content categories.

        Use this when you need to categorize content
        for advertising or content taxonomy.

        Args:
            text (str): Input text to classify.

        Returns:
            Dict[str, Any]: A JSON object mapping IAB
                categories to confidence scores.
        """

        input_data = text

        response = self._run(
            model="zlm-v1-iab-classify-edge",
            input_data=input_data,
        )

        if "error" in response:
            return {"error": response["error"]}

        content = self._extract_text(response)

        try:
            return json.loads(content)

        except Exception:
            return {"error": "Failed to parse model output", "raw": content}

    def zerogpu_classify_iab_enriched(self, text: str) -> Dict[str, Any]:
        r"""Classify text into enriched IAB content categories.

        Use this when you need more detailed classification signals, including
        additional metadata beyond standard IAB categories.

        Args:
            text (str): Input text to classify.

        Returns:
            Dict[str, Any]: A JSON object containing enriched category
            labels and additional metadata.
        """

        input_data = text

        response = self._run(
            model="zlm-v1-iab-classify-edge-enriched",
            input_data=input_data,
        )

        if "error" in response:
            return {"error": response["error"]}

        content = self._extract_text(response)

        try:
            return json.loads(content)

        except Exception:
            return {"error": "Failed to parse model output", "raw": content}

    def zerogpu_classify_zero_shot(
        self, text: str, labels: List[str]
    ) -> Dict[str, Any]:
        r"""Classify text into user-defined
        categories (zero-shot classification).

        Use this when predefined categories are required and do not follow
        a fixed taxonomy like IAB.

        Args:
            text (str): Input text to classify.
            labels (List[str]): List of candidate labels.

        Returns:
            Dict[str, Any]: A JSON object mapping labels to confidence scores.
        """

        input_data = text

        response = self._run(
            model="deberta-v3-small",
            input_data=input_data,
            extra={"categories": labels},
        )

        if "error" in response:
            return {"error": response["error"]}

        content = self._extract_text(response)

        try:
            return json.loads(content)

        except Exception:
            return {"error": "Failed to parse model output", "raw": content}

    def zerogpu_classify_structured(
        self,
        text: str,
        schema_definition: Dict[str, List[str]],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        r"""Structured classification using schema.

        Use this when you need to classify text into
        predefined structured fields.

        Args:
            text (str): The text to classify.
            schema_definition (Dict[str, List[str]]):
                The schema for structured classification.
            threshold (Optional[float], optional):
                The threshold for classification. Defaults to None.

        Returns:
            Dict[str, Any]: The classification results.
        """

        metadata: Dict[str, Any] = {
            "task": "structured-classification",
            "schema": schema_definition,
        }

        if threshold is not None:
            metadata["threshold"] = threshold

        input_data = text

        response = self._run(
            model="gliner2-base-v1",
            input_data=input_data,
            extra={"metadata": metadata},
        )

        if "error" in response:
            return {"error": response["error"]}

        content = self._extract_text(response)

        try:
            return json.loads(content)

        except Exception:
            return {"error": "Failed to parse", "raw": content}

    def zerogpu_extract_entities(
        self,
        text: str,
        labels: List[str],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        r"""Extract named entities from text using specified labels.

        Use this when you need to identify entities such as PERSON,
        ORGANIZATION, or LOCATION from text.

        Args:
            text (str): The text to extract entities from.
            labels (List[str]):
                The list of labels to use for entity extraction.
            threshold (Optional[float], optional):
                The threshold for entity extraction. Defaults to None.

        Returns:
            Dict[str, Any]: The entity extraction results.
        """

        metadata: Dict[str, Any] = {
            "task": "entity-extraction",
            "labels": labels,
        }

        if threshold is not None:
            metadata["threshold"] = threshold

        input_data = text

        response = self._run(
            model="gliner2-base-v1",
            input_data=input_data,
            extra={"metadata": metadata},
        )

        if "error" in response:
            return {"error": response["error"]}

        content = self._extract_text(response)

        try:
            return json.loads(content)

        except Exception:
            return {"error": "Failed to parse", "raw": content}

    def zerogpu_extract_json(
        self,
        text: str,
        schema: Dict[str, List[str]],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        r"""Extract structured JSON from text using a provided schema.

        Use this when you want to map unstructured
        text into a structured format.

        Args:
            text (str): Input text.
            schema (Dict[str, List[str]]): Field definitions for extraction.
            threshold (Optional[float]): Confidence threshold.

        Returns:
            Dict[str, Any]: Extracted structured data matching the schema.
        """

        metadata: Dict[str, Any] = {
            "task": "json-extraction",
            "schema": schema,
        }

        if threshold is not None:
            metadata["threshold"] = threshold

        input_data = text

        response = self._run(
            model="gliner2-base-v1",
            input_data=input_data,
            extra={"metadata": metadata},
        )

        if "error" in response:
            return {"error": response["error"]}

        content = self._extract_text(response)

        try:
            return json.loads(content)

        except Exception:
            return {"error": "Failed to parse", "raw": content}

    def zerogpu_extract_pii(
        self,
        text: str,
        threshold: Optional[float] = None,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        r"""Extract PII entities from text.

        Use this when you need to identify sensitive information
        such as names, emails, or phone numbers in text.

        Args:
            text (str): The text to extract PII from.
            threshold (Optional[float]): Confidence threshold.
            categories (Optional[List[str]], optional):
                The list of PII categories to extract. Defaults to None.

        Returns:
            Dict[str, Any]: The PII extraction results.
        """

        metadata: Dict[str, Any] = {"usecase": "extract"}

        if categories:
            metadata["categories"] = categories

        if threshold is not None:
            metadata["threshold"] = threshold

        input_data = text

        response = self._run(
            model="gliner-multi-pii-v1",
            input_data=input_data,
            extra={"metadata": metadata},
        )

        if "error" in response:
            return {"error": response["error"]}

        content = self._extract_text(response)

        try:
            return json.loads(content)

        except Exception:
            return {"error": "Failed to parse model output", "raw": content}

    def zerogpu_redact_pii(
        self,
        text: str,
        mask: str = "label",
        threshold: Optional[float] = None,
        categories: Optional[List[str]] = None,
    ) -> str:
        r"""Redact personally identifiable information (PII) from text.

        Use this when sensitive information such as names, emails, or phone
        numbers must be masked.

        Args:
            text (str): Input text.
            mask (str): Redaction strategy (e.g., "label").
            threshold (Optional[float]): Confidence threshold.
            categories (Optional[List[str]]): Specific PII types to redact.

        Returns:
            str: Text with PII replaced according to the mask strategy.
        """

        metadata: Dict[str, Any] = {
            "usecase": "redact",
            "mask": mask,
        }

        if categories:
            metadata["categories"] = categories

        if threshold is not None:
            metadata["threshold"] = threshold

        input_data = text

        response = self._run(
            model="gliner-multi-pii-v1",
            input_data=input_data,
            extra={"metadata": metadata},
        )

        if "error" in response:
            return f"Error: {response['error']}"

        content = self._extract_text(response)

        try:
            parsed = json.loads(content)
            return parsed.get("redacted_text", "")

        except Exception:
            return content

    def _run(
        self,
        model: str,
        input_data: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        r"""Execute a ZeroGPU model request via SDK.

        Args:
            model (str): Model identifier.
            input_data (Any): Input payload.
            extra (Optional[Dict[str, Any]]): Additional parameters.

        Returns:
            Dict[str, Any]: Raw response or error dict.
        """

        payload = {
            "model": model,
            "input": input_data,
            "text": {"format": {"type": "text"}},
        }

        if extra:
            for k, v in extra.items():
                payload[k] = v

        logger.debug(f"ZeroGPU request payload: {payload}")

        try:
            return self.client.responses.create_response(**payload)

        except Exception as e:
            logger.error(f"ZeroGPU request failed: {e}")
            return {"error": str(e)}

    def _tool(self, func):
        return FunctionTool(func, synthesize_schema=False)

    def get_tools(self) -> List[FunctionTool]:
        return [
            self._tool(self.zerogpu_chat),
            self._tool(self.zerogpu_chat_thinking),
            self._tool(self.zerogpu_summarize),
            self._tool(self.zerogpu_classify_iab),
            self._tool(self.zerogpu_classify_iab_enriched),
            self._tool(self.zerogpu_classify_zero_shot),
            self._tool(self.zerogpu_classify_structured),
            self._tool(self.zerogpu_extract_entities),
            self._tool(self.zerogpu_extract_json),
            self._tool(self.zerogpu_extract_pii),
            self._tool(self.zerogpu_redact_pii),
        ]

    def _extract_text(self, response: Dict[str, Any]) -> str:
        r"""Extract the text from the API response.

        Args:
            response (Dict[str, Any]): The API response.

        Returns:
            str: The extracted text.
        """
        try:
            if hasattr(response, "output"):
                return response.output[0].content[0].text
            return response["output"][0]["content"][0]["text"]

        except Exception as e:
            logger.warning(f"Failed to extract text: {e}")
            return ""
