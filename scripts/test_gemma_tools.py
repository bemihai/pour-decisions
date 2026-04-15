"""Phase 0 validation script for Gemma 4 / Ollama integration.

Runs three checks:
1. Basic inference -- model responds to a plain prompt.
2. Tool calling -- model correctly invokes a tool with arguments.
3. Structured output -- model returns a valid Pydantic model.

Usage:
    PYTHONPATH=. python3 scripts/test_gemma_tools.py [--model MODEL] [--base-url URL]

Expected model: gemma4:e2b (5.1B Q4_K_M) running via Ollama on localhost:11434.

Note: Gemma 4 uses an internal reasoning/thinking pass before generating the final
response, so do NOT set a tight num_predict limit -- it will cut off the thinking
tokens and produce an empty content string.
"""

import argparse
import sys
import time

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Gemma 4 / Ollama capabilities")
    parser.add_argument("--model", default="gemma4:e2b", help="Ollama model tag")
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama server URL")
    return parser.parse_args()


def _make_llm(model: str, base_url: str, **kwargs) -> ChatOllama:
    # Google-recommended sampling parameters for Gemma 4
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Check 1: Basic inference
# ---------------------------------------------------------------------------

def check_basic_inference(model: str, base_url: str) -> bool:
    """Verify the model responds to a plain text prompt.

    Note: Do not set num_predict here -- Gemma 4 uses an internal reasoning pass
    whose tokens consume the budget before visible output is produced.
    """
    print("\n--- Check 1: Basic inference ---")
    llm = _make_llm(model, base_url)
    start = time.time()
    try:
        result = llm.invoke("Name one famous Bordeaux chateau in five words or less.")
        elapsed = time.time() - start
        content = result.content if hasattr(result, "content") else str(result)
        print(f"  Response : {content!r}")
        print(f"  Latency  : {elapsed:.1f}s")
        ok = bool(content.strip())
        print(f"  {PASS if ok else FAIL}")
        return ok
    except Exception as e:
        print(f"  {FAIL} -- {e}")
        return False


# ---------------------------------------------------------------------------
# Check 2: Tool calling
# ---------------------------------------------------------------------------

@tool
def get_cellar_wines(wine_type: str = "") -> str:
    """Get wines from the user's cellar, optionally filtered by type.

    Args:
        wine_type: Optional wine type filter, e.g. 'red', 'white', 'rose'.

    Returns:
        A text summary of matching cellar wines.
    """
    return f"Found 3 bottles of Barolo 2018 (red) and 2 bottles of Chablis 2021 (white)."


def check_tool_calling(model: str, base_url: str) -> bool:
    """Verify the model emits a structured tool call."""
    print("\n--- Check 2: Tool calling ---")
    llm = _make_llm(model, base_url)
    llm_with_tools = llm.bind_tools([get_cellar_wines])
    start = time.time()
    try:
        result = llm_with_tools.invoke("What red wines do I have in my cellar?")
        elapsed = time.time() - start
        tool_calls = getattr(result, "tool_calls", [])
        print(f"  Tool calls : {tool_calls}")
        print(f"  Content    : {result.content!r}")
        print(f"  Latency    : {elapsed:.1f}s")
        ok = len(tool_calls) > 0 and tool_calls[0].get("name") == "get_cellar_wines"
        print(f"  {PASS if ok else FAIL} (tool_calls found: {len(tool_calls)})")
        if not ok:
            print("  NOTE: Model did not call a tool. Hybrid strategy may be needed (see Phase 4.2).")
        return ok
    except Exception as e:
        print(f"  {FAIL} -- {e}")
        return False


# ---------------------------------------------------------------------------
# Check 3: Structured output
# ---------------------------------------------------------------------------

class WineAnalysis(BaseModel):
    """Structured wine analysis returned by the LLM."""

    description: str = Field(description="2-3 sentence description of the wine's flavor profile and style")
    drink_from_year: int | None = Field(None, description="Year the wine begins drinking well")
    drink_to_year: int | None = Field(None, description="Year the wine is past its peak")


def check_structured_output(model: str, base_url: str) -> bool:
    """Verify the model can return a validated Pydantic model via with_structured_output."""
    print("\n--- Check 3: Structured output ---")
    llm = _make_llm(model, base_url)
    structured_llm = llm.with_structured_output(WineAnalysis)
    start = time.time()
    try:
        result = structured_llm.invoke(
            "Describe a 2018 Barolo from Giacomo Conterno in Piedmont, Italy."
        )
        elapsed = time.time() - start
        print(f"  description     : {result.description!r}")
        print(f"  drink_from_year : {result.drink_from_year}")
        print(f"  drink_to_year   : {result.drink_to_year}")
        print(f"  Latency         : {elapsed:.1f}s")
        ok = isinstance(result, WineAnalysis) and bool(result.description.strip())
        print(f"  {PASS if ok else FAIL}")
        return ok
    except Exception as e:
        print(f"  {FAIL} -- {e}")
        print("  NOTE: Structured output failed. Description service should use cloud model (Phase 5.2).")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    model = args.model
    base_url = args.base_url

    print(f"Gemma 4 / Ollama validation  --  model={model}  url={base_url}")
    print("=" * 60)

    results = {
        "basic_inference": check_basic_inference(model, base_url),
        "tool_calling": check_tool_calling(model, base_url),
        "structured_output": check_structured_output(model, base_url),
    }

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results.items():
        status = PASS if passed else FAIL
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if results["basic_inference"]:
        print("Basic inference works -- local model is usable.")
    else:
        print("Basic inference FAILED -- check that Ollama is running and model is pulled.")

    if results["tool_calling"]:
        print("Tool calling works -- intelligent agent can use Gemma 4 directly.")
    else:
        print("Tool calling FAILED -- intelligent agent will need the hybrid strategy (Phase 4.2).")

    if results["structured_output"]:
        print("Structured output works -- description service can use local model.")
    else:
        print("Structured output FAILED -- description service should use cloud model (Phase 5.2).")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()






