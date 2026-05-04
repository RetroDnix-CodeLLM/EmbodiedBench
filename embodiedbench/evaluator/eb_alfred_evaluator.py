import os
import numpy as np
from tqdm import tqdm
import time
import json
from embodiedbench.envs.eb_alfred.EBAlfEnv import EBAlfEnv, ValidEvalSets
from embodiedbench.planner.vlm_planner import VLMPlanner
from embodiedbench.evaluator.summarize_result import average_json_values
from embodiedbench.evaluator.evaluator_utils import load_saved_data, update_config_with_args
from embodiedbench.evaluator.config.system_prompts import alfred_system_prompt
from embodiedbench.main import logger

example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_examples.json')
react_example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_react_examples.json')
exploration_example_path = os.path.join(os.path.dirname(__file__), 'config/alfred_long_horizon_examples.json')
system_prompt = alfred_system_prompt

def encode_base64(image_path):
    import base64
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode("utf-8")
    return encoded_string

from openai import OpenAI
client = OpenAI(
    api_key="sk-8306544ef471453db48dc7f88f61dc82",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
eocv_input_tokens = 0
eocv_output_tokens = 0
eocv_cached_tokens = 0
EOBS_PROMPT_SINGLE_FRAME = """
The image shows the state of a virtual environment after executing an action '{action_str}', check if the observation is consistent with the expected textual observation '{eobs_single}'. 

First, describe the environment state and reason carefully, then gives your answer(choose between yes or no) in <answer></answer> tag. 

Possible actions: 
• Find: Parameterized by the name of the receptacle to navigate to. So long as the object is present in the scene, this skill is always valid
• Pick up: Parameterized by the name of the object to pick. Only valid if the robot is close to the object, not holding another object, and the object is not inside a closed receptacle.
• Put down: Parameterized by the name of the object to put down to a nearby receptacle. Only valid if the robot is holding an object.
• Drop: Parameterized by the name of the object to put down. It is different from Put down action, as this does not guarantee the held object will be put into a specified receptacle. 
• Open: Parameterized by the name of the receptacle to open. Only valid if the receptacle is closed and the robot is close to the receptacle.
• Close: Parameterized by the name of the receptacle to close. Only valid if the receptacle is open and the robot is close to the receptacle.
• Turn on: Parameterized by the name of the object to turn on. Only valid if the object is turned off and the robot is close to the object.
• Turn off: Parameterized by the name of the object to turn off. Only valid if the object is turned on and the robot is close to the object.
• Slice: Parameterized by the name of the object to slice. Only valid if the object is sliceable and the robot is close to the object.

Note: 
the action can be invalid and failed, this way you will likely see no change in the environment before and after the action, and in this case the correct answer should be no.
"""
def validate_eobs(action_str, eobs_single, last_frame_path, current_frame_path):
    global eocv_input_tokens, eocv_output_tokens, eocv_cached_tokens
    message = [
        {
            "role": "user",
            "content": [
                # {
                #     "type": "image_url",
                #     "image_url": {
                #         "url": f"data:image/jpeg;base64,{encode_base64(last_frame_path)}"
                #     },
                # },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_base64(current_frame_path)}"
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
        model="qwen3.6-flash",
        messages=message,
        max_tokens=2048,
        # temperature=0.0,
        # top_p=0.95,
        # presence_penalty=0.0,
        extra_body={
            # "repetition_penalty": 1.0,
            # "top_k": 20,
            # "min_p": 0.0,
            "enable_thinking": False,
        }
    )
    if usage := getattr(chat_response, "usage", None):
        eocv_input_tokens += usage.prompt_tokens
        eocv_output_tokens += usage.completion_tokens
        if usage.prompt_tokens_details is not None and isinstance(usage.prompt_tokens_details.cached_tokens, int):
            eocv_cached_tokens += usage.prompt_tokens_details.cached_tokens
    try:
        # response_text = chat_response.choices[0].message.reasoning
        response_text = chat_response.choices[0].message.content
        if "<answer>" not in response_text or "</answer>" not in response_text:
            # Fallback behavior
            if 'yes' in response_text.lower():
                answer = "yes"
            else:
                answer = "no"
        else:
            answer = response_text.split("<answer>")[1].split("</answer>")[0].strip().lower()
    except Exception as e:
        response_text = str(chat_response) + str(e)
        answer = "no"
    return answer, response_text

class EB_AlfredEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None

    def check_config_valid(self):
        if self.config['multistep'] + self.config['chat_history'] > 1:
            raise ValueError("Only one of multistep, chat_history can be enabled at a time.")
        
        if self.config['language_only']:
            if self.config['multistep']:
                logger.warning("Language only mode should not have multistep enabled. Setting these arguments to False ...")
                self.config['multistep'] = 0
        
    def save_episode_metric(self, episode_info, eocv_info):
        episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        filename = 'episode_{}_final_res.json'.format(episode_idx)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)
        
        if os.getenv("EXTRA_EOCV"):
            eocv_res_path = os.path.join(self.env.log_path, 'eocv_results')
            if not os.path.exists(eocv_res_path):
                os.makedirs(eocv_res_path)
            eocv_fn = 'episode_{}_eocv.json'.format(self.env._current_episode_num)
            with open(os.path.join(eocv_res_path, eocv_fn), 'w', encoding='utf-8') as f:
                json.dump(eocv_info, f, indent=4, ensure_ascii=False)
    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        valid_eval_sets = list(valid_eval_sets)
        if type(valid_eval_sets) == list and len(valid_eval_sets) == 0:
            valid_eval_sets = ValidEvalSets

        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"
            self.env = EBAlfEnv(eval_set=self.eval_set, down_sample_ratio=self.config['down_sample_ratio'], 
                                          exp_name=exp_name, selected_indexes=self.config.get('selected_indexes', []), 
                                          detection_box=self.config.get('detection_box', False),
                                          resolution=self.config.get('resolution', 500), 
                                          )
            examples = json.load(open(react_example_path, 'r+')) if os.getenv("EXTRA_ONE_STEP") else json.load(open(example_path, 'r+')) if self.eval_set != 'long_horizon' else json.load(open(exploration_example_path, 'r+'))
            model_type = self.config.get('model_type', 'remote')
            self.planner = VLMPlanner(self.model_name, model_type, self.env.language_skill_set, system_prompt, examples, n_shot=self.config['n_shots'], 
                                            obs_key='head_rgb', chat_history=self.config['chat_history'], language_only=self.config['language_only'],
                                            use_feedback=self.config.get('env_feedback', True), multistep=self.config.get('multistep', 0), tp=self.config.get('tp', 1))

            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), output_file='summary.json')
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            last_obs_path = img_path
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            # update the action space for alfred due to dynamic objects
            self.planner.set_actions(self.env.language_skill_set)
            done = False
            eocv_info = []
            while not done:
                try: 
                    action, reasoning, eobs = self.planner.act(img_path, user_instruction)
                    if os.getenv("EXTRA_EOCV") and eobs is not None:
                        print(f"Planner Expected Observation: {eobs}")
                    if action == -2: # empty plan stop here
                        if len(self.planner.episode_act_feedback) > 0:
                            self.planner.episode_act_feedback[-1][-1] += "Task is not completed yet, try to figure out the problem and avoid output empty plan."
                        else:
                            episode_info['empty_plan'] = 1
                            self.env.episode_log.append({
                                'last_action_success': 0.0,
                                'action_id': -2,
                                'action_description': 'empty plan',
                                'reasoning': reasoning,
                            })
                            info = {
                                'task_success': episode_info.get('task_success', 0),
                                'task_progress': episode_info.get("task_progress", 0),
                                'env_step': self.env._current_step,
                            }
                            break 
                    if action == -1:
                        self.env._cur_invalid_actions += 1
                        episode_info['reward'].append(-1)
                        episode_info['num_invalid_actions'] += 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -1,
                            'action_description': 'invalid action',
                            'reasoning': reasoning,
                        })
                        info = {
                            'task_success': episode_info.get('task_success', 0),
                            'task_progress': episode_info.get("task_progress", 0),
                            'env_step': self.env._current_step,
                        }
                        if self.env._cur_invalid_actions >= self.env._max_invalid_actions:
                            break
                        continue
                    if isinstance(action, list):
                        readable_planner_action = [self.env.language_skill_set[action_id] if type(action_id) == int and action_id >= 0 else action_id for action_id in action]
                        print(f"Planner Output Action: {readable_planner_action}")
                    # mutiple actions
                    if type(action) == list:
                        if os.getenv("EXTRA_MULTI_STEP"):
                            action_lim = 5
                        elif os.getenv("EXTRA_ONE_STEP"):
                            if len(action) > 0 and isinstance(action[0], int) and self.env.language_skill_set[action[0]].startswith("find"):
                                action_lim = 2
                            else:
                                action_lim = 1
                        else:
                            action_lim = 1000
                        action_length = min(self.env._max_episode_steps - self.env._current_step, len(action), action_lim)
                        for idx, action_single in enumerate(action[:action_length]):
                            obs, reward, done, info = self.env.step(action_single, reasoning=reasoning)
                            action_str = action_single if type(action_single) == str else self.env.language_skill_set[action_single]
                            print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                            if info['task_success']:
                                if action_str.startswith("pick up"):
                                    info['env_feedback'] = "Last action executed successfully. The item you picked up is now shown in the center bottom of the observation image."
                                elif action_str.startswith("put down"):
                                    info['env_feedback'] = "Last action executed successfully. The item in your hand is now put down to the closest available receptacle. Hint: use 'find recep' to target at the receptacle before putting down the object in hand to it. "
                            logger.debug(f"reward: {reward}")
                            logger.debug(f"terminate: {done}\n")
                            self.planner.update_info(info)
                            img_path = self.env.save_image(obs)
                            episode_info['reward'].append(reward)
                            episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                            
                            if os.getenv("EXTRA_EOCV") and eobs is not None and idx < len(eobs) and not action_str.lower().startswith("find") and not idx == action_length - 1: # only validate non-navigate actions and not the last action which is more likely to be affected by the following actions
                                eobs_single = eobs[idx]
                                answer, full_output = validate_eobs(action_str, eobs_single, last_obs_path, img_path)
                                eocv_info.append({
                                    'step': self.env._current_step,
                                    'action': action_str,
                                    'observation': img_path,
                                    'feedback': info['env_feedback'] if 'env_feedback' in info else '',
                                    'eobs': eobs_single,
                                    'eocv_answer': answer,
                                    'eocv_full_output': full_output,
                                    'eocv_pass': answer == "yes",
                                })
                                if answer != "yes":
                                    print("EOCV check failed. Replanning ...")
                                    break
                            last_obs_path = img_path
                            # Only stop when done
                            # if done or not info['last_action_success']:
                            if done:
                                # stop or replanning
                                print("Invalid action or task complete. If invalid then Replanning.")
                                break
                    else: # single action
                        obs, reward, done, info = self.env.step(action, reasoning=reasoning)
                        action_str = action if type(action) == str else self.env.language_skill_set[action]
                        print(f"Executed action: {action_str}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        
                        self.planner.update_info(info)
                        img_path = self.env.save_image(obs)
                        episode_info['reward'].append(reward)
                        episode_info['num_invalid_actions'] += (info['last_action_success'] == 0)
                
                except Exception as e: 
                    # Print complete stack back trace
                    import traceback
                    traceback.print_exc()
                    logger.error(f"Error in planner.act: {e}")
                    time.sleep(30)

            
            temp_usage_path = os.path.join(self.env.log_path, 'results')
            os.makedirs(temp_usage_path, exist_ok=True)
            json.dump({
                    'input_tokens': self.planner.model.input_tokens,
                    'output_tokens': self.planner.model.output_tokens,
                    'cached_tokens': self.planner.model.cached_tokens,
                }, 
                open(os.path.join(temp_usage_path, f'total_token_usage.json'), 'w'), 
                ensure_ascii=False, 
                indent=4
            )
            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            episode_info["task_progress"] = info['task_progress']
            episode_info['num_steps'] = info["env_step"]
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["num_invalid_actions"] = episode_info['num_invalid_actions']
            episode_info["num_invalid_action_ratio"] = episode_info['num_invalid_actions'] / info["env_step"] if info['env_step'] > 0 else 0
            episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", time.time() - self.env._episode_start_time)

            self.env.save_episode_log()
            self.save_episode_metric(episode_info, eocv_info)
            progress_bar.update()


if __name__ == '__main__':
    import argparse
    def parse_arguments():
        parser = argparse.ArgumentParser(description='Change configuration parameters.')
        parser.add_argument('--model_name', type=str, help='Name of the model.')
        parser.add_argument('--n_shots', type=int, help='Number of examples')
        parser.add_argument('--down_sample_ratio', type=float, help='Down sample ratio.')
        parser.add_argument('--model_type', type=str, help='Type of the model.')
        parser.add_argument('--language_only', type=int, help='Set to True for language only mode.')
        parser.add_argument('--exp_name', type=str, help='Name of the experiment.')
        parser.add_argument('--chat_history', type=int, help='Set to True to enable chat history.')
        parser.add_argument('--detection_box', type=int, help='Set to True to enable detection.')
        parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
        parser.add_argument('--multistep', type=int, help='Number of steps for multi-step reasoning.')
        parser.add_argument('--resolution', type=int, help='Resolution for processing.')
        parser.add_argument('--env_feedback', type=int, help='Set to True to enable environment feedback.')
        parser.add_argument('--tp', type=int, help='number of tensor parallel splits of the model parameters')
        return parser.parse_args()


    config = {
        'model_name': 'gpt-4o-mini', # 'Qwen/Qwen2-VL-7B-Instruct',
        'n_shots': 10,
        'down_sample_ratio': 1.0,
        'model_type': 'remote', # 'local', 
        'language_only': 0,
        'exp_name': 'vlm_10shots_imgsize500',
        'chat_history': 0, 
        'detection_box': 0,
        'eval_sets': ['base'], 
        'selected_indexes': [], 
        'multistep':0, 
        'resolution': 500, 
        'env_feedback': 1,
        'tp': 1,
    }

    args = parse_arguments()
    update_config_with_args(config, args)

    evaluator = EB_AlfredEvaluator(config)
    evaluator.evaluate_main()




