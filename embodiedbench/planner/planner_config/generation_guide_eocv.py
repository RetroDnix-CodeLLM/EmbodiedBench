vlm_generation_guide_eocv={
    "type": "object",
    "additionalProperties": False,
    'properties': {
        "visual_state_description": {
            "type": "string",
            "description": "Description of current state from the visual image",
        },
        "reasoning_and_reflection": {
            "type": "string",
            "description": "summarize the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task",
        },
        "language_plan": {
            "type": "string",
            "description": "The list of actions to achieve the user instruction. Each action is started by the step number and the action name",
        },
        "executable_plan": {
            "type": "array",
            "description": "A list of actions needed to achieve the user instruction, with each action having an action ID and a name. Do not output empty list.",
            "items": {
                "type": "object",
                "properties": {
                    "action_id": {
                        "type": "integer",
                        "description": "The action ID to select from the available actions given by the prompt",
                    },
                    "action_name": {
                        "type": "string",
                        "description": "The name of the action",
                    },
                    "expected_observation": {
                        "type": "string",
                        "description": "The expected observation(described in text) after executing the action, which is used to verify if the plan is correct.",
                    }
                },
                "required": ["action_id", "action_name", "expected_observation"],
            }
        },
    },
    "required": ["visual_state_description", "reasoning_and_reflection", "language_plan", "executable_plan"]
}
