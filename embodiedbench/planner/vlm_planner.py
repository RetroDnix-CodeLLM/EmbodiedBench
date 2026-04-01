import torch
import re
import os
import time
import numpy as np
import cv2
import json
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url, template, template_lang, fix_json
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.main import logger

class VLMPlanner():
    def __init__(self, model_name, model_type, actions, system_prompt, examples, n_shot=0, obs_key='head_rgb', 
                chat_history=False, language_only=False, use_feedback=True, multistep=0, tp=1, kwargs={}):
        self.model_name = model_name
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.examples = examples
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.set_actions(actions)
        self.model_type = model_type
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

        self.use_feedback = use_feedback
        self.multistep = multistep
        self.planner_steps = 0
        self.output_json_error = 0
        self.language_only = language_only
        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')
    
    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str


    def process_prompt(self, user_instruction, prev_act_feedback=[]):
        user_instruction = user_instruction.rstrip('.')
        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            if self.language_only:
                prompt += f" You are supposed to output in json. You need to output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
            else:
                prompt += f" You are supposed to output in json. You need to describe current visual state from the image, output your reasoning steps and plan. At the end, output the action id (0 ~ {len(self.actions)-1}) from the available actions to excute."
                if os.getenv("EXTRA_MULTI_STEP") != None:
                    prompt += "You should generate at most 5 actions in the plan."
                elif os.getenv("EXTRA_ONE_STEP") != None:
                    prompt += "You are required to generate only 1 action for the plan. Reflection on the current state and your previous actions, decide the best action to take at the current step, and do not output more than 1 action."
                elif os.getenv("EXTRA_EOCV") != None:
                    examples = [
                        "Example 1:\nHuman instruction: Put both an toy airplane and a bowl onto the black table.\nOutput: {'language_plan': 'To achieve the goal, the robot must locate the toy airplane and the bowl in the room and then move each item to the black table. There are two tables in the room, and the robot identifies table 1 as the black table. Therefore, the objective is to place both objects on table 1. The plan is as follows: first, navigate to the sofa, pick up the airplane, move to table 1, and place the airplane there. Then, proceed to table 2, where the bowl might be, pick up the bowl, return to table 1, and set the bowl there.', 'executable_plan': [{'action_id': 12, 'action_name': 'navigate to the sofa', 'expected_observation': 'The robot is near the sofa.'}, {'action_id': 47, 'action_name': 'pick up the toy airplane', 'expected_observation': 'The toy airplane is picked up by the robot.'}, {'action_id': 6, 'action_description': 'navigate to the table 1', 'expected_observation': 'The robot is near the table 1.'}, {'action_id': 50, 'action_description': 'place at the table 1', 'expected_observation': 'The toy airplane is placed on the table 1.'}, {'action_id': 7, 'action_description': 'navigate to the table 2', 'expected_observation': 'The robot is near the table 2.'}, {'action_id': 42, 'action_description': 'pick up the bowl', 'expected_observation': 'The bowl is picked up by the robot.'}, {'action_id': 6, 'action_description': 'navigate to the table 1', 'expected_observation': 'The robot is near the table 1.'}, {'action_id': 50, 'action_description': 'place at the table 1', 'expected_observation': 'The bowl is placed on the table 1.'}]}\n\n",
                        "Example 2:\nHuman instruction: I made a mistake and left the fridge open. Can you assist me by closing it?\nOutput: {'language_plan': 'The objective is for the robot to close the refrigerator. To do so, the robot first navigates to the refrigerator and then closes it.', 'executable_plan': [{'action_id': 13, 'action_name': 'navigate to the refrigerator', 'expected_observation': 'The robot is near the refrigerator.'}, {'action_id': 61, 'action_name': 'close the refrigerator', 'expected_observation': 'The refrigerator is closed.'}]}\n\n"
                    ] 
                    prompt += "You should generate at most 5 actions in the plan. Additionally, in order to verify if your plan is executable, you need to output your expected observation(discribed in text) after executing each action in the plan. The expected visual state should be concise and only include key information such as object locations and states, without detailed descriptions. Here are some examples:\n\n" + examples[0] + examples[1]
        
        elif self.chat_history:
            prompt = f'The human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the executable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
        else:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples[:self.n_shot])])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')
            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'
            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                if self.use_feedback:
                    prompt += '\nStep {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
                else:
                    prompt += '\nStep {}, action id {}, {}'.format(i, action_feedback[0], self.actions[action_feedback[0]])

            if self.language_only:
                prompt += f'''\n\n Considering the above interaction history, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
            else:
                prompt += f'''\n\n Considering the above interaction history and the current image state, to achieve the human instruction: '{user_instruction}', you are supposed to output in json. You need to describe current visual state from the image, summarize interaction history {'and environment feedback ' if self.use_feedback else ''}and reason why the last action or plan failed and did not finish the task, output your new plan to achieve the goal from current state. At the end, output the excutable plan with action ids(0 ~ {len(self.actions)-1}) from the available actions.'''
                if os.getenv("EXTRA_MULTI_STEP") != None:
                    prompt += "You should generate at most 5 actions in the plan."
                elif os.getenv("EXTRA_ONE_STEP") != None:
                    prompt += "Unlike the examples, you are required to generate only 1 action for the plan."
                elif os.getenv("EXTRA_EOCV") != None:
                    examples = [
                        "Example 1:\nHuman instruction: Put both an toy airplane and a bowl onto the black table.\nOutput: {'language_plan': 'To achieve the goal, the robot must locate the toy airplane and the bowl in the room and then move each item to the black table. There are two tables in the room, and the robot identifies table 1 as the black table. Therefore, the objective is to place both objects on table 1. The plan is as follows: first, navigate to the sofa, pick up the airplane, move to table 1, and place the airplane there. Then, proceed to table 2, where the bowl might be, pick up the bowl, return to table 1, and set the bowl there.', 'executable_plan': [{'action_id': 12, 'action_name': 'navigate to the sofa', 'expected_observation': 'The robot is near the sofa.'}, {'action_id': 47, 'action_name': 'pick up the toy airplane', 'expected_observation': 'The toy airplane is picked up by the robot.'}, {'action_id': 6, 'action_description': 'navigate to the table 1', 'expected_observation': 'The robot is near the table 1.'}, {'action_id': 50, 'action_description': 'place at the table 1', 'expected_observation': 'The toy airplane is placed on the table 1.'}, {'action_id': 7, 'action_description': 'navigate to the table 2', 'expected_observation': 'The robot is near the table 2.'}, {'action_id': 42, 'action_description': 'pick up the bowl', 'expected_observation': 'The bowl is picked up by the robot.'}, {'action_id': 6, 'action_description': 'navigate to the table 1', 'expected_observation': 'The robot is near the table 1.'}, {'action_id': 50, 'action_description': 'place at the table 1', 'expected_observation': 'The bowl is placed on the table 1.'}]}\n\n",
                        "Example 2:\nHuman instruction: I made a mistake and left the fridge open. Can you assist me by closing it?\nOutput: {'language_plan': 'The objective is for the robot to close the refrigerator. To do so, the robot first navigates to the refrigerator and then closes it.', 'executable_plan': [{'action_id': 13, 'action_name': 'navigate to the refrigerator', 'expected_observation': 'The robot is near the refrigerator.'}, {'action_id': 61, 'action_name': 'close the refrigerator', 'expected_observation': 'The refrigerator is closed.'}]}\n\n"
                    ] 
                    prompt += "You should generate at most 5 actions in the plan. Additionally, in order to verify if your plan is correct, you need to output your expected observation(discribed in text) after executing each action in the plan. Here are some examples:\n\n" + examples[0] + examples[1]
        # print(prompt)
        return prompt
    

    def get_message(self, image, prompt, messages=[]):
        if self.language_only:
            return messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}],
                }
            ]
        else:
            if type(image) == str:
                image_path = image 
            else:
                image_path = './evaluation/tmp_{}.png'.format(len(messages)//2)
                cv2.imwrite(image_path, image)

            if self.multistep: # handle multiple images
                ind = int(image_path.split('step_')[-1].strip('.png'))
                content = [{"type": "text", "text": prompt}]
                for i in range(max(ind - self.multistep + 1, 0), ind +1):
                    temp_path = ''.join(image_path.split('step_')[:-1])+ f'step_{str(i)}.png'
                    temp_data_url = local_image_to_data_url(image_path=temp_path)
                    content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": temp_data_url,
                            }})
            else:
                data_url = local_image_to_data_url(image_path=image_path)
                content = [{ "type": "image_url", "image_url": { "url": data_url,}}, {"type": "text", "text": prompt}]

            return messages + [
                {
                    "role": "user",
                    "content": content,
                }
            ]

    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action
    
    def json_to_action(self, output_text, json_key='executable_plan'):
        eobs = None
        try:
            json_object = json.loads(output_text)
            action = [x[self.action_key] for x in json_object[json_key]]
            eobs = [x.get('expected_observation', '') for x in json_object[json_key]]
            if not len(action):
                print('empty plan, stop here')
                action = -2
            else:
                # keep action valid
                for i, act in enumerate(action):
                    if act >= len(self.actions) or act < 0:
                        print('found invlid action')
                        if i == 0:
                            action = -1
                        else:
                            action = action[:i]
                        break
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            self.output_json_error += 1
            action = -1
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            self.output_json_error += 1
            action = -1
        return action, eobs

    
        
    def act_custom(self, prompt, obs):
        assert type(obs) == str # input image path
        out = self.model.respond(prompt, obs)
        # fix common generated json errors
        out = fix_json(out)
        logger.debug(f"Model Output:\n{out}\n")
        action, eobs = self.json_to_action(out)
        self.planner_steps += 1
        return action, out, eobs


    def act(self, observation, user_instruction):
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path
        
        prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        # some models do not support json scheme, add style into prompt
        if 'claude' in self.model_name or 'InternVL' in self.model_name or 'Qwen2-VL' in self.model_name or 'Qwen2.5-VL' in self.model_name or self.model_type == 'custom':
            prompt = prompt + template_lang if self.language_only else prompt + template

        if self.model_type == 'custom':
            return self.act_custom(prompt, obs) 

        if len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        for entry in self.episode_messages:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        if 'gemini-1.5-pro' in self.model_name or 'gemini-2.0-flash' in self.model_name:
            try: 
                out = self.model.respond(self.episode_messages)
                time.sleep(15)
            except Exception as e:
                print("An unexpected error occurred:", e)
                time.sleep(60)
                out = self.model.respond(self.episode_messages)
        else:
            try: 
                out = self.model.respond(self.episode_messages)
            except Exception as e:
                print("An unexpected error occurred:", e)

                if self.model_type != 'local':
                    time.sleep(60)
                else:
                    time.sleep(20)
                out = self.model.respond(self.episode_messages)
        logger.debug(f"Model Output:\n{out}\n")

        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
        action, eobs = self.json_to_action(out)
        self.planner_steps += 1
        return action, out, eobs

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])


        

