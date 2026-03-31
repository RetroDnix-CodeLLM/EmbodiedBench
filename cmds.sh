
# Openrouter
export OPENAI_API_KEY="sk-S9iP8pIdbK8XD4Hojm9g5rx1Rano3Muo8tsTFexIT6oJCuRV"

EXTRA_MULTI_STEP=1 python -m embodiedbench.main env=eb-alf model_name=gpt-5-mini exp_name='baseline'


export DASHSCOPE_API_KEY="sk-ae6624a5b29848ed87132c9c7e8a375c"

EXTRA_MULTI_STEP=1 python -m embodiedbench.main env=eb-alf model_name="qwen3.5-plus-2026-02-15" exp_name="native"