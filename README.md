# Conversational API Agent with LangChain

This repository contains a Python project that demonstrates how to build a conversational agent using LangChain to interact with APIs. The agent is capable of fetching current temperature data using the Open-Meteo API and retrieving summaries from Wikipedia.

## Project Setup

### Prerequisites

- Python 3.8+
- pip
- An OpenAI API key

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/conversational-agent-langchain.git
   cd conversational-agent-langchain
   ```

2. **Install dependencies:**

   ```bash
   pip install openai requests pydantic wikipedia langchain-community dotenv
   ```

3. **Environment Configuration:**

   Create a `.env` file in the project root and add your OpenAI API key:

   ```plaintext
   OPENAI_API_KEY='your_openai_api_key_here'
   ```

   Load the environment variables:

   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

### Usage

To run the agent, execute the main script:

```bash
python main.py
```

This will start an interactive session where you can ask the agent about current temperatures or to summarize Wikipedia articles.

## Project Structure

- `main.py`: Contains the main script to run the conversational agent.
- `.env`: Stores configuration variables and API keys.

## Tools and Functions

### Defining Tools

Tools are defined to perform specific API calls. Here's how they are set up:

1. **Current Temperature Tool:**

   Fetches the current temperature for specified coordinates using the Open-Meteo API.

   ```python
   from langchain.agents import tool
   from pydantic import BaseModel, Field

   class OpenMeteoInput(BaseModel):
       latitude: float = Field(...)
       longitude: float = Field(...)

   @tool(args_schema=OpenMeteoInput)
   def get_current_temperature(latitude: float, longitude: float) -> str:
       # Implementation here
   ```

2. **Wikipedia Search Tool:**

   Searches Wikipedia and provides summaries of the top articles related to the query.

   ```python
   @tool
   def search_wikipedia(query: str) -> str:
       # Implementation here
   ```

### Running the Agent

The agent uses a chain of operations including input parsing, tool execution, and response generation managed by LangChain's infrastructure.

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor

# Set up the chat model and prompt template
# Define the agent chain and executor
# Interactive agent function to handle user inputs
```

This README provides a clear guide on how to set up, run, and understand the project, ensuring that anyone checking the repository can get started with minimal setup and understand the architecture and functionality of the conversational agent. Adjust paths, URLs, and specific instructions according to your actual project structure and external API usage.
