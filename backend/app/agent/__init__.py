"""Agent package: LLM providers, prompts, bounded bundles, MCP tool surface.

All engine->LLM traffic goes through `llm.LLMClient` (metered, cached,
retrying). `tools` and `bundle` are read-only over the graph.
"""
