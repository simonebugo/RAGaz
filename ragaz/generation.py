"""Stage 8: configure a generator independently of the retrieval pipeline."""

import os

from langchain_openai import ChatOpenAI

from ragaz.config import LOCAL_BASE_URL, LOCAL_MODEL_NAME, OPENAI_MODEL_NAME


def build_local_llm() -> ChatOpenAI:
    """Create a client for Qwen served by LM Studio on the configured host.

    Creating a client does not generate text. invoke() sends the messages to the
    server. Despite the class name, this configuration connects to LM Studio.
    """
    return ChatOpenAI(
        model=os.getenv("LM_STUDIO_MODEL") or LOCAL_MODEL_NAME,
        base_url=os.getenv("LM_STUDIO_BASE_URL") or LOCAL_BASE_URL,
        # The placeholder works without server authentication. If enabled in
        # LM Studio, provide its actual token through LM_STUDIO_API_KEY.
        api_key=os.getenv("LM_STUDIO_API_KEY") or "lm-studio",
        temperature=0.2,
    )


def build_openai_llm() -> ChatOpenAI:
    """Create the optional cloud client using OPENAI_API_KEY from the environment.

    This provider requires API credit. It uses the same prompt and retrieved
    passages as the local provider, not a different embedding model.
    The project's successful end-to-end generation has been verified with Qwen.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment or .env file.")

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL") or OPENAI_MODEL_NAME,
        base_url="https://api.openai.com/v1",
        reasoning_effort="none",
        max_completion_tokens=800,
        use_responses_api=False,
    )
