
# Openrouter
export OPENAI_API_KEY="sk-S9iP8pIdbK8XD4Hojm9g5rx1Rano3Muo8tsTFexIT6oJCuRV"

EXTRA_MULTI_STEP=1 python -m embodiedbench.main env=eb-alf model_name=gpt-5-mini exp_name='baseline'


export DASHSCOPE_API_KEY="sk-8306544ef471453db48dc7f88f61dc82"

EXTRA_MULTI_STEP=1 python -m embodiedbench.main env=eb-alf model_name="qwen3.5-plus-2026-02-15" exp_name="native"

EXTRA_EOCV=1 python -m embodiedbench.main env=eb-hab model_name=qwen3-vl-plus exp_name='eocv'