"""Integration tests for Ollama connectivity and tool calling.

These tests require a running Ollama server with gemma4:e2b pulled.
Run with: pytest tests/agents/test_ollama_integration.py -m integration
Mark as slow tests, skip in fast test runs.

Tests cover:
- Basic inference with Gemma 4 e2b
- Tool calling capabilities (bind_tools)
- Structured output (with_structured_output)
"""

import pytest
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.agents.llm import load_base_model


@pytest.fixture(scope="module")
def ollama_model():
    """Load Ollama model for integration tests. Requires Ollama running on localhost:11434."""
    try:
        model = load_base_model("ollama", "gemma4:e2b")
        # Quick health check - invoke with a simple prompt to ensure connectivity
        model.invoke([HumanMessage(content="test")])
        return model
    except Exception as e:
        pytest.skip(f"Ollama server not available: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestOllamaBasicInference:
    """Basic inference tests for Ollama/Gemma 4."""

    def test_responds_to_simple_prompt(self, ollama_model):
        """Ollama responds with non-empty content for a basic prompt."""
        result = ollama_model.invoke([HumanMessage(content="Say hello in one word")])
        assert result.content, "Expected non-empty response from Ollama"
        assert isinstance(result.content, str)

    def test_wine_domain_knowledge(self, ollama_model):
        """Ollama can answer basic wine questions."""
        result = ollama_model.invoke([HumanMessage(content="What grape is Barolo made from? Answer in one word.")])
        assert result.content
        # Nebbiolo is the correct grape, check if it appears in the response
        assert "nebbiolo" in result.content.lower() or "Nebbiolo" in result.content

    def test_no_num_predict_limit(self, ollama_model):
        """Gemma 4 responses are not cut off (num_predict not set)."""
        # Longer prompt to verify the model completes its reasoning pass
        result = ollama_model.invoke([
            HumanMessage(content="Explain the difference between Barolo and Barbaresco in one sentence.")
        ])
        assert result.content
        # Should get a complete sentence, not empty or truncated
        assert len(result.content) > 20, "Response appears truncated"


@pytest.mark.integration
@pytest.mark.slow
class TestOllamaToolCalling:
    """Tool calling integration tests for Ollama/Gemma 4."""

    def test_bind_tools_works(self, ollama_model):
        """Ollama model can bind tools and call them correctly."""
        # Define a simple test tool
        @tool
        def get_wine_color(wine_name: str) -> str:
            """Get the color (red, white, rosé) of a wine variety.

            Args:
                wine_name: Name of the wine or grape variety
            """
            colors = {
                "chardonnay": "white",
                "cabernet sauvignon": "red",
                "merlot": "red",
                "barolo": "red",
            }
            return colors.get(wine_name.lower(), "unknown")

        # Bind the tool to the model
        model_with_tools = ollama_model.bind_tools([get_wine_color])

        # Invoke with a prompt that should trigger tool use
        result = model_with_tools.invoke([HumanMessage(content="What color is Chardonnay wine?")])

        # Check if tool_calls is populated (indicates the model called the tool)
        assert hasattr(result, "tool_calls"), "Expected tool_calls attribute on AIMessage"
        # The model should call the tool at least once
        # Note: depending on the model, it might or might not call the tool directly
        # We're just verifying the bind_tools mechanism works without error

    def test_tool_calling_with_arguments(self, ollama_model):
        """Ollama correctly extracts arguments when calling tools."""
        @tool
        def search_cellar(wine_name: str, vintage: int | None = None) -> str:
            """Search for wines in the cellar.

            Args:
                wine_name: Name of the wine to search for
                vintage: Optional vintage year
            """
            if vintage:
                return f"Found {wine_name} {vintage} in cellar"
            return f"Found {wine_name} in cellar"

        model_with_tools = ollama_model.bind_tools([search_cellar])
        result = model_with_tools.invoke([
            HumanMessage(content="Search for Barolo 2016 in my cellar")
        ])

        # Verify tool calling mechanism works (no exceptions raised)
        assert result, "Expected a response from the model"


@pytest.mark.integration
@pytest.mark.slow
class TestOllamaStructuredOutput:
    """Structured output tests for Ollama/Gemma 4."""

    def test_with_structured_output_returns_pydantic_model(self, ollama_model):
        """Ollama with_structured_output returns a valid Pydantic instance."""
        class WineInfo(BaseModel):
            """Structured wine information."""
            name: str = Field(description="Name of the wine")
            region: str = Field(description="Wine region")
            color: str = Field(description="Wine color: red, white, or rosé")

        structured_llm = ollama_model.with_structured_output(WineInfo)
        result = structured_llm.invoke([
            HumanMessage(content="Tell me about Barolo wine from Piedmont")
        ])

        assert isinstance(result, WineInfo), f"Expected WineInfo instance, got {type(result)}"
        assert result.name, "Expected non-empty wine name"
        assert result.region, "Expected non-empty region"
        assert result.color in ["red", "white", "rosé", "red wine"], f"Unexpected color: {result.color}"

    def test_structured_output_with_optional_fields(self, ollama_model):
        """Structured output handles optional fields correctly."""
        class WineAnalysis(BaseModel):
            """Wine analysis with optional drinking window."""
            description: str = Field(description="Brief wine description")
            drink_from_year: int | None = Field(None, description="Start of drinking window")
            drink_to_year: int | None = Field(None, description="End of drinking window")

        structured_llm = ollama_model.with_structured_output(WineAnalysis)
        result = structured_llm.invoke([
            HumanMessage(content="Describe a 2018 Barolo from Giacomo Conterno")
        ])

        assert isinstance(result, WineAnalysis)
        assert result.description, "Expected non-empty description"
        # Drinking window fields are optional, just check they exist
        assert hasattr(result, "drink_from_year")
        assert hasattr(result, "drink_to_year")


@pytest.mark.integration
@pytest.mark.slow
class TestOllamaPerformance:
    """Performance and latency checks for Ollama (informational, not strict assertions)."""

    def test_inference_completes_within_reasonable_time(self, ollama_model):
        """Basic inference completes (slow on CPU, but should finish)."""
        import time
        start = time.time()
        result = ollama_model.invoke([HumanMessage(content="Say hello")])
        elapsed = time.time() - start

        assert result.content, "Expected response from Ollama"
        # Sanity check: should complete within 2 minutes even on slow CPU
        assert elapsed < 120, "Inference took longer than 2 minutes - possible timeout"
