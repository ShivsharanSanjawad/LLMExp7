class BackendUnavailable(Exception):
    """Raised when the LLM server (LM Studio) or Qdrant cannot be reached or returns an error."""

    def __init__(self, component: str):
        super().__init__(component)
        self.component = component  # "llm" or "qdrant"
