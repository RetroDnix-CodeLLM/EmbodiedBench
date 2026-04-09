from openai import OpenAI


def encode_base64(image_path):
    import base64

    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
    return encoded_string


client = OpenAI(
    api_key="sk-ae6624a5b29848ed87132c9c7e8a375c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
eocv_input_tokens = 0
eocv_output_tokens = 0
eocv_cached_tokens = 0

EOBS_PROMPT_DUOBLE_FRAME = """
The first image shows the original environment state, while the second image shows the environment state after executing an action '{action_str}', check if the observation is consistent with the expected textual observation '{eobs_single}'. 

First, describe the environment state and reason carefully, then gives your answer(choose between yes or no) in <answer></answer> tag. 

Possible actions: 
1. Pick: picks up object from nearby locations. You should focus on the suction cup gripper of the robot arm. If the intend object is isolated or not present in the first frame, and close to the arm in the second frame, you can decide the object is picked up.
2. Place: places object in a specified location. You should focus on the specified location. If the location is present in the frame, and something is placed to the location in the second frame, you can decide the object is placed.
3. Open: open the specific receptacle. You should focus on the state of the receptacle. If the door of the receptacle is closed in the first frame, and open in the second frame, you can decide the receptacle is opened.
4. Close: close the specific receptacle. You should focus on the state of the receptacle. If the door of the receptacle is open in the first frame, and closed in the second frame, you can decide the receptacle is closed.

Note: 
the action can be invalid and failed, this way you will likely see no change in the environment before and after the action, and in this case the correct answer should be no.
"""

EOBS_PROMPT_SINGLE_FRAME = """
The given image is the environment observation after executing action '{action_str}', check if the observation is consistent with the expected textual observation '{eobs_single}'. 

First, describe the environment state and reason carefully, then gives your answer(choose between yes or no) in <answer></answer> tag.

Possible actions: 
1. Pick: picks up specified object from nearby locations. You should focus on the suction cup gripper of the robot arm. If the intend object is close to the arm in the frame, you can decide the object is picked up.
2. Place: places the holding object in a specified location. You should focus on specified location. If the location is presentd in the frame, you can decide the object is placed.
3. Open: open the specific receptacle. You should focus on the state of the receptacle. If the door of the receptacle is open in the frame, you can decide the receptacle is opened.
4. Close: close the specific receptacle. You should focus on the state of the receptacle. If the door of the receptacle is closed in the frame, you can decide the receptacle is closed.

Note: 
the action can be invalid and failed, this way you will likely see no change in the environment before and after the action, and in this case the correct answer should be no."""


def validate_eobs(action_str, eobs_single, frame_path):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_base64(frame_path)}"
                    },
                },
                {
                    "type": "text",
                    "text": EOBS_PROMPT_SINGLE_FRAME.format(
                        action_str=action_str, eobs_single=eobs_single
                    ),
                },
            ],
        }
    ]
    chat_response = client.chat.completions.create(
        # model="/home/hyzheng2/QYProjects/models/Qwen/Qwen3.5-2B",
        model="qwen3-vl-plus",
        messages=message,
        max_tokens=2048,
        # temperature=0.6,
        # top_p=0.95,
        # presence_penalty=0.0,
        # extra_body={
        #     "repetition_penalty": 1.0,
        #     "top_k": 20,
        #     "min_p": 0.0,
        #     "enable_thinking": True,
        # }
    )

    response_text = chat_response.choices[0].message.content
    print(response_text)


def validate_eobs_2(action_str, eobs_single, last_frame_path, current_frame_path):
    message = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_base64(last_frame_path)}"
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_base64(current_frame_path)}"
                    },
                },
                {
                    "type": "text",
                    "text": EOBS_PROMPT_DUOBLE_FRAME.format(
                        action_str=action_str, eobs_single=eobs_single
                    ),
                },
            ],
        }
    ]
    chat_response = client.chat.completions.create(
        # model="/home/hyzheng2/QYProjects/models/Qwen/Qwen3.5-2B",
        model="qwen3-vl-plus",
        messages=message,
        max_tokens=2048,
        # temperature=0.6,
        # top_p=0.95,
        # presence_penalty=0.0,
        # extra_body={
        #     "repetition_penalty": 1.0,
        #     "top_k": 20,
        #     "min_p": 0.0,
        #     "enable_thinking": True,
        # }
    )

    response_text = chat_response.choices[0].message.content
    print(response_text)


if __name__ == "__main__":
    test_cases = [
        {
            "name": "Pick up the orange",
            "action_str": "pick up the orange",
            "eobs_single": "The orange is picked up by the agent.",
            "last_frame_path": "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0409-010146_ord/base/images/episode_2/episode_2_step_1.png",
            "current_frame_path": "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0409-010146_ord/base/images/episode_2/episode_2_step_2.png",
        },
        {
            "name": "Place in the sink",
            "action_str": "place in the sink",
            "eobs_single": "The orange is placed at the sink in the kitchen.",
            "last_frame_path": "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0409-010146_ord/base/images/episode_2/episode_2_step_3.png",
            "current_frame_path": "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0409-010146_ord/base/images/episode_2/episode_2_step_4.png",
        },
        {
            "name": "Open the fridge",
            "action_str": "open the fridge",
            "eobs_single": "The fridge is opened.",
            "last_frame_path": "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0409-010146_ord/base/images/episode_4/episode_4_step_5.png",
            "current_frame_path": "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0409-010146_ord/base/images/episode_4/episode_4_step_6.png",
        },
        {
            "name": "Close the fridge",
            "action_str": "close the fridge",
            "eobs_single": "The fridge is closed.",
            "last_frame_path": "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0409-010146_ord/base/images/episode_6/episode_6_step_1.png",
            "current_frame_path": "/home/hyzheng2/QYProjects/EmbodiedBench/running/eb_habitat/qwen3-vl-plus_eocv-0409-010146_ord/base/images/episode_6/episode_6_step_2.png",
        }
    ]
    for case in test_cases:
        print(case['name'])
        validate_eobs(case['action_str'], case['eobs_single'], case['current_frame_path'])
        print("-" * 50)
        validate_eobs_2(
            case['action_str'], case['eobs_single'], case['last_frame_path'], case['current_frame_path']
        )
        print("\n\n")
