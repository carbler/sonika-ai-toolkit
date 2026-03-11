# Getting Started

## Installation

```bash
pip install sonika-ai-toolkit
```

## API Keys

Create a `.env` file in your project root:

```env
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here       # optional
GOOGLE_API_KEY=your_gemini_key_here            # optional
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_key_here # optional
AWS_REGION=us-east-1                           # optional
```

## Your First Agent

```python
import os
from dotenv import load_dotenv
from sonika_ai_toolkit import OrchestratorBot, OpenAILanguageModel
from sonika_ai_toolkit.tools.integrations import EmailTool

load_dotenv()

llm = OpenAILanguageModel(os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini")
bot = OrchestratorBot(
    strong_model=llm,
    fast_model=llm,
    instructions="You are a communications assistant.",
    tools=[EmailTool()],
    memory_path="/tmp/bot_memory",
)

result = bot.run("Send a hello email to user@example.com")
print(result.content)
print("Tools used:", [t["tool_name"] for t in result.tools_executed])
```

## Your First Classifier

```python
import os
from sonika_ai_toolkit import SentimentClassifier, OpenAILanguageModel

llm = OpenAILanguageModel(os.getenv("OPENAI_API_KEY"))
classifier = SentimentClassifier(llm=llm)

result = classifier.classify("I love this product!")
print(result.result)
# {'sentiment': 'positive', 'confidence': 0.95, 'reasoning': '...'}
```

## Async Streaming

```python
import asyncio
from sonika_ai_toolkit import OrchestratorBot, OpenAILanguageModel

async def main():
    llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
    bot = OrchestratorBot(
        strong_model=llm, fast_model=llm,
        instructions="You are a helpful assistant.",
        tools=[],
        memory_path="/tmp/bot_memory",
    )

    async for stream_mode, payload in bot.astream_events("Hello!", mode="auto"):
        if stream_mode == "updates":
            for node_name, update in payload.items():
                if node_name == "agent" and update.get("final_report"):
                    print("Result:", update["final_report"])

asyncio.run(main())
```
