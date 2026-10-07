RESEARCH_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "string",
                        "enum": [
                            "tavily",
                            "arxiv",
                            "wikipedia",
                        ],
                    },
                    "query": {
                        "type": "string",
                    },
                },
                "required": [
                    "tool",
                    "query",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "steps",
    ],
    "additionalProperties": False,
}