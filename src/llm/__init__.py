"""LLM package — provider abstraction for all LLM interactions."""
from src.llm.interface import LLMInterface, LLMError
from src.llm.mock_llm import MockLLM

__all__ = ["LLMInterface", "LLMError", "MockLLM"]
