# Classifiers

Sonika AI Toolkit provides five classifiers for structured text and image classification. All classifiers support both sync (`classify`) and async (`aclassify`) methods, and return `ClassificationResponse` with token usage tracking.

## ClassificationResponse

All classifiers return:

```python
class ClassificationResponse(BaseModel):
    input_tokens: int
    output_tokens: int
    result: dict[str, Any]
```

## TextClassifier

The base classifier — define any custom schema with Pydantic:

```python
from pydantic import BaseModel, Field
from sonika_ai_toolkit import TextClassifier, OpenAILanguageModel

class TicketClassification(BaseModel):
    category: str = Field(..., description="The ticket category")
    priority: str = Field(..., description="Priority: low, medium, high, critical")
    language: str = Field(..., description="Detected language code (e.g. en, es)")

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
classifier = TextClassifier(llm=llm, validation_class=TicketClassification)

result = classifier.classify("My server is down and customers can't access the app!")
print(result.result)
# {'category': 'infrastructure', 'priority': 'critical', 'language': 'en'}
```

## IntentClassifier

Classifies text into predefined intents with confidence scores and entity extraction:

```python
from sonika_ai_toolkit import IntentClassifier, OpenAILanguageModel

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
classifier = IntentClassifier(
    llm=llm,
    intents=["book_flight", "cancel_booking", "check_status", "general_inquiry"],
    descriptions={
        "book_flight": "User wants to book a new flight",
        "cancel_booking": "User wants to cancel an existing booking",
        "check_status": "User wants to check flight or booking status",
    },
)

result = classifier.classify("I need to fly from NYC to London next Friday")
print(result.result)
# {'intent': 'book_flight', 'confidence': 0.95, 'entities': {'origin': 'NYC', 'destination': 'London', 'date': 'next Friday'}}
```

**Output schema:**

| Field | Type | Description |
|-------|------|-------------|
| `intent` | `str` | One of the provided intents |
| `confidence` | `float` | 0.0 – 1.0 confidence score |
| `entities` | `dict` | Extracted entities |

## SentimentClassifier

Zero-config sentiment analysis — no schema needed:

```python
from sonika_ai_toolkit import SentimentClassifier, OpenAILanguageModel

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
classifier = SentimentClassifier(llm=llm)

result = classifier.classify("This product exceeded all my expectations!")
print(result.result)
# {'sentiment': 'positive', 'confidence': 0.92, 'reasoning': 'The text expresses strong satisfaction...'}
```

**Output schema:**

| Field | Type | Description |
|-------|------|-------------|
| `sentiment` | `str` | `positive`, `negative`, `neutral`, or `mixed` |
| `confidence` | `float` | 0.0 – 1.0 confidence score |
| `reasoning` | `str` | Brief explanation |

## SafetyClassifier

Content safety moderation with customizable categories:

```python
from sonika_ai_toolkit import SafetyClassifier, OpenAILanguageModel

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
classifier = SafetyClassifier(llm=llm)

result = classifier.classify("I love sunny days at the park.")
print(result.result)
# {'is_safe': True, 'categories': [], 'severity': 'none', 'explanation': '...'}
```

**Default safety categories:** `hate_speech`, `violence`, `sexual_content`, `self_harm`, `pii`, `harassment`, `illegal_activity`

**Custom categories:**

```python
classifier = SafetyClassifier(
    llm=llm,
    custom_categories=["misinformation", "spam", "phishing"],
)
```

**Output schema:**

| Field | Type | Description |
|-------|------|-------------|
| `is_safe` | `bool` | Whether the text is safe |
| `categories` | `list[str]` | Flagged categories |
| `severity` | `str` | `none`, `low`, `medium`, or `high` |
| `explanation` | `str` | Safety assessment explanation |

## ImageClassifier

Multimodal image classification using vision-capable LLMs. Supports URLs and local files.

```python
from pydantic import BaseModel, Field
from sonika_ai_toolkit import ImageClassifier, OpenAILanguageModel

class SceneAnalysis(BaseModel):
    description: str = Field(..., description="Brief description of the image")
    objects: list[str] = Field(..., description="Main objects detected")
    mood: str = Field(..., description="Overall mood or atmosphere")

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
classifier = ImageClassifier(llm=llm, validation_class=SceneAnalysis)

# From URL
result = classifier.classify("https://example.com/photo.jpg")
print(result.result)

# From local file
result = classifier.classify("/path/to/image.png")

# With additional context
result = classifier.classify("photo.jpg", context="This is from a security camera")
```

!!! note "Supported models"
    ImageClassifier requires a vision-capable LLM: **OpenAI** (gpt-4o, gpt-4o-mini) or **Gemini** (all models).

## Async Usage

All classifiers support async:

```python
import asyncio

async def main():
    classifier = SentimentClassifier(llm=llm)
    result = await classifier.aclassify("Great service!")
    print(result.result)

asyncio.run(main())
```

## Top-Level Imports

```python
from sonika_ai_toolkit import (
    TextClassifier,
    ClassificationResponse,
    IntentClassifier,
    SentimentClassifier,
    SafetyClassifier,
    ImageClassifier,
)
```
