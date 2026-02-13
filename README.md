# Sonika AI Toolkit <a href="https://pepy.tech/projects/sonika-ai-toolkit"><img src="https://static.pepy.tech/badge/sonika-ai-toolkit" alt="PyPI Downloads"></a>

A robust Python library designed to build state-of-the-art conversational agents and AI tools. It leverages `LangChain` and `LangGraph` to create autonomous bots capable of complex reasoning and tool execution.

## Installation

```bash
pip install sonika-ai-toolkit
```

## Prerequisites

You'll need the following API keys depending on the model you wish to use:

- OpenAI API Key
- DeepSeek API Key (Optional)
- Google Gemini API Key (Optional)
- AWS Bedrock API Key (Optional, for Bedrock)

Create a `.env` file in the root of your project with the following variables:

```env
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
GOOGLE_API_KEY=your_gemini_key_here
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_api_key_here
AWS_REGION=us-east-1
```

## Key Features

- **Multi-Model Support**: Agnostic integration with OpenAI, DeepSeek, Google Gemini, and Amazon Bedrock.
- **Conversational Agent**: Robust agent (`ReactBot`) with native tool execution capabilities and LangGraph state management.
- **Tasker Agent**: Advanced planner-executor agent (`TaskerBot`) for complex multi-step tasks.
- **Structured Classification**: Text classification with strongly typed outputs.
- **Document Processing**: Utilities for processing PDFs, DOCX, and other formats with intelligent chunking.
- **Custom Tools**: Easy integration of custom tools via Pydantic and LangChain.

## Basic Usage

### Conversational Agent with Tools

```python
import os
from dotenv import load_dotenv
from sonika_ai_toolkit.tools.integrations import EmailTool
from sonika_ai_toolkit.agents.react import ReactBot
from sonika_ai_toolkit.utilities.types import Message
from sonika_ai_toolkit.utilities.models import OpenAILanguageModel

# Load environment variables
load_dotenv()

# Configure model
api_key = os.getenv("OPENAI_API_KEY")
language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini', temperature=0.7)

# Configure tools
tools = [EmailTool()]

# Create agent instance
bot = ReactBot(language_model, instructions="You are a helpful assistant", tools=tools)

# Get response
user_message = 'Send an email to erley@gmail.com saying hello'
messages = [Message(content="My name is Erley", is_bot=False)]
response = bot.get_response(user_message, messages, logs=[])

print(response["content"])
```

### Text Classification

```python
import os
from sonika_ai_toolkit.classifiers.text import TextClassifier
from sonika_ai_toolkit.utilities.models import OpenAILanguageModel
from pydantic import BaseModel, Field

# Define classification structure
class Classification(BaseModel):
    intention: str = Field()
    sentiment: str = Field(..., enum=["happy", "neutral", "sad", "excited"])

# Initialize classifier
model = OpenAILanguageModel(os.getenv("OPENAI_API_KEY"))
classifier = TextClassifier(llm=model, validation_class=Classification)

# Classify text
result = classifier.classify("I am very happy today!")
print(result.result)
```

## Available Components

### Agents
- **ReactBot**: Standard agent using LangGraph workflow.
- **TaskerBot**: Advanced planner agent for multi-step tasks.

### Utilities
- **ILanguageModel**: Unified interface for LLM providers.
- **DocumentProcessor**: Text extraction and chunking utilities.

## Project Structure

```
src/sonika_ai_toolkit/
├── agents/             # Bot implementations
├── classifiers/        # Text classification tools
├── document_processing/# PDF and document tools
├── tools/             # Tool definitions
└── utilities/         # Models and common types
```

## License

This project is licensed under the MIT License.
