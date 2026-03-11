# Models

Sonika AI Toolkit supports four LLM providers. All implement the `ILanguageModel` interface with `predict()`, `invoke()`, and `stream_response()` methods.

## OpenAI

```python
from sonika_ai_toolkit import OpenAILanguageModel

llm = OpenAILanguageModel(
    api_key="sk-...",
    model_name="gpt-4o-mini",  # default
    temperature=0.7,
)
```

**Supported models:** `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1`, `o1-mini`, etc.

## Google Gemini

```python
from sonika_ai_toolkit import GeminiLanguageModel

llm = GeminiLanguageModel(
    api_key="...",
    model_name="gemini-2.5-flash",
    temperature=0.7,
)
```

!!! warning "Thinking models"
    Gemini thinking models (`gemini-2.5-*`, `*-thinking`, `*thinking-exp*`) require `temperature=1.0` — this is automatically overridden with a warning.

**Thinking model behavior:** When `include_thoughts=True`, `response.content` is a list containing thinking and text blocks. The toolkit handles this automatically.

## DeepSeek

```python
from sonika_ai_toolkit import DeepSeekLanguageModel

llm = DeepSeekLanguageModel(
    api_key="...",
    model_name="deepseek-chat",  # or "deepseek-reasoner"
)
```

!!! note "DeepSeek Reasoner"
    `deepseek-reasoner` (and models with `r1` in the name) uses a custom `_DeepSeekReasonerChatModel` that captures `reasoning_content`. It does **not** support tool calling — a `ValueError` is raised if tools are provided.

## Amazon Bedrock

```python
from sonika_ai_toolkit import BedrockLanguageModel

llm = BedrockLanguageModel(
    api_key="...",
    model_name="amazon.nova-micro-v1:0",
    region="us-east-1",
)
```

The `AWS_BEARER_TOKEN_BEDROCK` env var is set automatically during initialization.

## ILanguageModel Interface

All models implement:

```python
from sonika_ai_toolkit import ILanguageModel

class ILanguageModel(ABC):
    model: BaseChatModel  # underlying LangChain model

    def predict(self, prompt: str) -> str: ...
    def invoke(self, prompt: str) -> ResponseModel: ...
    def stream_response(self, prompt: str) -> Iterator: ...
```

Type-hint against `ILanguageModel` for provider-agnostic code:

```python
def classify(llm: ILanguageModel, text: str):
    return llm.predict(f"Classify: {text}")
```

## Top-Level Imports

```python
from sonika_ai_toolkit import (
    OpenAILanguageModel,
    GeminiLanguageModel,
    DeepSeekLanguageModel,
    BedrockLanguageModel,
    ILanguageModel,
)
```
