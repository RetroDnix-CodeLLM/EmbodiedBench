from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv
import json

prompt = """
## You are a robot operating in a home. You are able to conduct the following actions:

## Action Descriptions and Validity Rules
• Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
• Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
• Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
• Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle. 
• Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
• Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
• Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
• Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
• Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.

Now, your task is to predict the observation after executing a given action in the current scene(given as image). Note that now the actions are randomly sampled and can be useless or invalid, if so, use "Nothing Happened" as its observation.

Your output should be in the following JSON format:
{
    "state_description": "Describe the current state of the given scene in detail",
    "reasoning": "Describe the reasoning process for how the action would affect the scene",
    "observation": "Describe the observation after executing the action. If the action is invalid or useless, output 'Nothing Happened'."
}
"""

response_format = {
    "type": "json_schema",
    "json_schema": {
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
            "state_description": {
                "type": "string",
                "description": "Detailed description of the current scene state before the action"
            },
            "reasoning": {
                "type": "string",
                "description": "Step-by-step reasoning about how the action affects the scene, including validity check"
            },
            "observation": {
                "type": "string",
                "description": "Resulting observation after executing the action, or 'Nothing Happened' if invalid/useless",
                "minLength": 1
            }
        },
        "required": ["state_description", "reasoning", "observation"],
        "additionalProperties": False
      }
    }
  }

from openai import OpenAI

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

from PIL import Image
import base64
from io import BytesIO

def pil_to_base64(img: Image.Image, format="PNG"):
    buffer = BytesIO()
    img.save(buffer, format=format)  # 写入内存
    byte_data = buffer.getvalue()   # 获取二进制
    base64_str = base64.b64encode(byte_data).decode("utf-8")
    return f"data:image/{format.lower()};base64,{base64_str}"

if __name__ == "__main__":
    """
    Example usage of the EBAlfEnv environment.
    Demonstrates environment interaction with random actions.
    """
    env = EBAlfEnv(eval_set='base', down_sample_ratio=1.0, selected_indexes=[])
    env.reset()
    print([(i, name) for i, name in enumerate(env.language_skill_set)])
    for _ in range(30):
        # Select  action
        action = env.action_space.sample()
        
        action = env.language_skill_set[action]
        
        img = Image.fromarray(env.env.last_event.frame)
        
        # Execute action
        obs, reward, done, info = env.step(action)
        print(reward, done, info)

        if info['last_action_success'] >= 1.0:

            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Action: {action}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": pil_to_base64(img)
                                }
                            }
                        ]
                    }
                ],
                response_format=response_format
            )
            response_text = response.choices[0].message.content
            print(response_text)
            response_json = json.loads(response_text)
            observation = response_json["observation"]

            print(action)
            print("Observation from LLM:", observation)
        
        # Optional rendering and image saving
        env.save_image()
        if done:
            break
    env.close()
