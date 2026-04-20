import os
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class LLM:
    def __init__(self):
        self._model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        price_input  = float(os.getenv("LLM_PRICE_INPUT",  "0.150"))
        price_output = float(os.getenv("LLM_PRICE_OUTPUT", "0.600"))

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        usage = response.usage
        cost = (usage.prompt_tokens * price_input +
                usage.completion_tokens * price_output) / 1_000_000

        return LLMResponse(
            content=response.choices[0].message.content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=round(cost, 8),
        )


llm = LLM()