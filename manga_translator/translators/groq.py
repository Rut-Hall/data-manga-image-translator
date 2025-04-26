import groq
import os
import json
from typing import List

from .common import CommonTranslator, MissingAPIKeyException
from .keys import GROQ_API_KEY, GROQ_MODEL

class GroqTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese', 'CHT': 'Traditional Chinese', 'CSY': 'Czech',
        'NLD': 'Dutch', 'ENG': 'English', 'FRA': 'French', 'DEU': 'German',
        'HUN': 'Hungarian', 'ITA': 'Italian', 'JPN': 'Japanese', 'KOR': 'Korean',
        'PLK': 'Polish', 'PTB': 'Portuguese', 'ROM': 'Romanian', 'RUS': 'Russian',
        'ESP': 'Spanish', 'TRK': 'Turkish', 'UKR': 'Ukrainian', 'VIN': 'Vietnamese',
        'CNR': 'Montenegrin', 'SRP': 'Serbian', 'HRV': 'Croatian', 'ARA': 'Arabic',
        'THA': 'Thai', 'IND': 'Indonesian'
    }

    _MAX_REQUESTS_PER_MINUTE = 200
    _TIMEOUT = 40
    _RETRY_ATTEMPTS = 5
    _MAX_TOKENS = 8192

    _CONTEXT_RETENTION = os.environ.get('CONTEXT_RETENTION', '').lower() == 'true'
    _CONFIG_KEY = 'groq'
    _MAX_CONTEXT = int(os.environ.get('CONTEXT_LENGTH', '20'))

    _CHAT_SYSTEM_TEMPLATE = (
        "You are a professional manga translation engine. Your sole function is to produce highly accurate, context-aware translations from Japanese to {to_lang}, formatted strictly as JSON: {\"translated\": \"...\"}.\n\n"
        "Analyze prior and current panels as an interconnected narrative. Consider speaker tone, implied relationships, and sequential dialogue to deliver the most accurate meaning possible.\n\n"
        "Obey these rules:\n"
        "1. Translate accurately with contextual precision—do not over-literalize nor over-localize.\n"
        "2. Preserve honorifics, Japanese names, and cultural expressions as-is (e.g., '-san', 'Senpai'). Do not convert them.\n"
        "3. Do not infer or assign gender unless explicitly stated. Default to neutral language or implicit phrasing.\n"
        "4. Proper names must follow standard Hepburn romanization and be preserved exactly as in the source (e.g., '弥生' → 'Yayoi').\n"
        "5. For ambiguous or slang terms, choose the most common conversational meaning unless context indicates otherwise. If uncertain, use phonetic transliteration.\n"
        "6. Preserve original meaning and nuance. Imperatives, questions, emotional tone, and slang must match intent.\n"
        "7. Do not summarize or explain. Do not include any output except: {\"translated\": \"...\"}\n"
        "8. Retain original onomatopoeia and sound effects unless context explicitly requires translation.\n"
        "9. Maintain a natural, anime-style cadence and tone when translating dialogue.\n"
        "10. Do not expand or compress the text significantly. Keep translation length close to the original where possible.\n\n"
        "Remember: You are a language model tuned specifically for manga. Your job is to make the reading experience smooth, authentic, and respectful to the source material.\n"
        "Translate now into {to_lang} and return only JSON."
    )

    _CHAT_SAMPLE = [
        (
            'Translate into English. Return the result in JSON format.\n'
            '{"untranslated": "<|1|>恥ずかしい… 目立ちたくない… 私が消えたい…\\n<|2|>きみ… 大丈夫⁉\\n<|3|>なんだこいつ 空気読めて ないのか…？"}\n'
        ),
        (
            '{"translated": "<|1|>So embarrassing… I don’t want to stand out… I wish I could disappear…\\n<|2|>Hey… Are you okay!?\\n<|3|>What’s with this person? Can’t they read the room…?"}'
        )
    ]

    def __init__(self, check_groq_key=True):
        super().__init__()
        self.client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        if not self.client.api_key and check_groq_key:
            raise MissingAPIKeyException('Please set the GROQ_API_KEY environment variable before using the Groq translator.')
        self.token_count = 0
        self.token_count_last = 0
        self.config = None
        self.model = GROQ_MODEL
        self.messages = [
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]}
        ]

    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', self._CHAT_SYSTEM_TEMPLATE)

    @property
    def chat_sample(self):
        return self._config_get('chat_sample', self._CHAT_SAMPLE)

    @property
    def temperature(self) -> float:
        return self._config_get('temperature', default=0.3)

    @property
    def top_p(self) -> float:
        return self._config_get('top_p', default=0.92)

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        results = []
        for prompt in queries:
            response = await self._request_translation(to_lang, prompt)
            # response is now a dict, extract the translated text
            results.append(response.get("translated", ""))
        self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return results

    async def _request_translation(self, to_lang: str, prompt: str) -> dict:
        # Build the prompt
        prompt_with_lang = (
            f"Translate the following text into {to_lang}. Return the result in JSON format.\n\n"
            f"{{\"untranslated\": \"{prompt}\"}}\n"
        )
        # Maintain context
        self.messages.append({'role': 'user', 'content': prompt_with_lang})
        if len(self.messages) > self._MAX_CONTEXT:
            self.messages = self.messages[-self._MAX_CONTEXT:]

        # System message
        system_msg = {'role': 'system', 'content': self.chat_system_template.format(to_lang=to_lang)}

        # API call
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[system_msg] + self.messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p,
            stop=["}"]
        )

        # Update token counts
        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        # Raw content
        raw = response.choices[0].message.content.strip()

        # Clean and parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: try to extract the translated value
            cleaned = raw.strip('{}\"')
            data = {"translated": cleaned}

        # Context retention logic
        if self._CONTEXT_RETENTION:
            self.messages.append({'role': 'assistant', 'content': raw})
        else:
            self.messages.pop()

        return data
