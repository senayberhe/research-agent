RESEARCH_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "tavily_search",
        "description": (
            "Search the web for current information, news, "
            "websites, industry information, and broad research."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "arxiv_search",
        "description": (
            "Search academic papers on scientific and "
            "technical topics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The academic search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "wikipedia_search",
        "description": (
            "Search Wikipedia for definitions, background, "
            "historical context, and foundational information."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The Wikipedia search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]