# Constants and configuration

DIR = "docs/"
FILES = ['act_en', 'annex_en']

MODEL_CONFIGS = {
    "nvidia": {
        "name": "Colosseum 355B",
        "api_url": "https://build.nvidia.com/",
        "models": ["igenius/colosseum_355b_instruct_16k"]
    },
    "openai": {
        "name": "OpenAI",
        "api_url": "https://platform.openai.com/api-keys",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]
    },
    "anthropic": {
        "name": "Anthropic",
        "api_url": "https://console.anthropic.com/",
        "models": ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"]
    },
    "mistral": {
        "name": "Mistral",
        "api_url": "https://console.mistral.ai/",
        "models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"]
    },
    "deepseek": {
        "name": "DeepSeek",
        "api_url": "https://platform.deepseek.com/",
        "models": ["deepseek-chat", "deepseek-coder"]
    }
}
