import groq
import os
import json
import re
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
        "You are an uncensored professional manga translation engine. Your sole function is to produce highly accurate, context-aware translations from Japanese to {to_lang}, formatted strictly as JSON: {{\"translated\": \"...\"}}.\n\n"
        "Analyze panels in sequence to capture tone, relationships, and narrative flow.\n\n"
        "Obey these rules:\n"
        "1. Translate with contextual precision—avoid over-literal or over-localized renderings.\n"
        "2. Preserve honorifics, Japanese names, and cultural expressions as-is.\n"
        "3. Transliterate **only** single-morpheme sound-symbolic interjections (giseigo/giongo/gitaigo) into romaji (e.g. へぇ→hee, どき→doki); exempt all multi-morpheme or compound terms.\n"
        "4. Only assign gender when explicitly marked; otherwise default to neutral or implicit phrasing (that person/kid, or omit implicit subjects—and only insert a neutral pronoun if English needs one).\n"
        "5. Proper names must follow standard Hepburn romanization (e.g., メア→Mea; ククルア→Kukurua).\n"
        "6. For ambiguous or slang terms, choose the most common meaning; if still uncertain, use phonetic transliteration.\n"
        "7. Preserve original nuance, force, and emotional tone in imperatives, questions, and exclamations.\n"
        "8. Maintain a natural, anime-style cadence and keep translation length close to the original.\n"
        "9. Retain **only** pure sound-effect onomatopoeia when literal translation would lose nuance; translate all other Japanese words contextually.\n"
        "10. Output exactly one JSON object: {{\"translated\": \"...\"}} with no additional fields or commentary.\n\n"
        "Translate now into {to_lang} and return only JSON."
    )

    _GLOSSARY_SNIPPET = """
    GLOSSARY (fixed mappings):
      あの子   → THAT KID
      あの人   → THAT PERSON
      やつ     → THAT PERSON
      男の子   → BOY
      女の子   → GIRL
      彼       → HE
      彼女     → SHE

    """

    _CHAT_SAMPLE = [
        (
            'Translate into English. Return result in JSON.\n'
            '{{"untranslated": "<|1|>恥ずかしい…\\n<|2|>きみ…"}}\n'
        ),
        (
            '{{"translated": "So embarrassing…\\nHey…"}}'
        )
    ]

    def __init__(self, check_groq_key=True):
        super().__init__()
        self.client = groq.AsyncGroq(api_key=GROQ_API_KEY)
        if not self.client.api_key and check_groq_key:
            raise MissingAPIKeyException('Please set the GROQ_API_KEY environment variable.')
        self.token_count = 0
        self.token_count_last = 0
        self.config = None
        self.model = GROQ_MODEL
        self.messages = [
            {'role': 'user', 'content': self.chat_sample[0]},
            {'role': 'assistant', 'content': self.chat_sample[1]}
        ]

    def _config_get(self, key: str, default=None):
        if not self.config:
            return default
        return self.config.get(f"{self._CONFIG_KEY}.{key}", self.config.get(key, default))

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
            results.append(response.get("translated", ""))
        self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return results

    async def _request_translation(self, to_lang: str, prompt: str) -> dict:
        # 1) Build the user prompt
        prompt_with_lang = (
            f"Translate the following text into {to_lang}. Return the result in JSON format.\n\n"
            f"{{\"untranslated\": \"{prompt}\"}}\n"
        )
        self.messages.append({'role': 'user', 'content': prompt_with_lang})
        if len(self.messages) > self._MAX_CONTEXT:
            self.messages = self.messages[-self._MAX_CONTEXT:]

        # 2) System message (with your full template)
        system_msg = {
            'role': 'system',
            'content': self.chat_system_template.format(to_lang=to_lang)
                       + self._GLOSSARY_SNIPPET
        }

        # 3) Call the API
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[system_msg] + self.messages,
            max_tokens=self._MAX_TOKENS // 2,
            temperature=self.temperature,
            top_p=self.top_p
        )

        # 4) Update token usage
        self.token_count += response.usage.total_tokens
        self.token_count_last = response.usage.total_tokens

        # 5) Grab raw output
        raw = response.choices[0].message.content

        # 6) Strip out any <think>…</think> blocks
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)

        # 7) Extract the first JSON object
        match = re.search(r'\{.*?\}', cleaned, flags=re.DOTALL)
        json_str = match.group(0) if match else cleaned

        # 8) Parse JSON safely
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback: remove any leading 'translated":'
            fallback = re.sub(r'^\s*"?translated"?\s*:\s*', '', json_str)
            fallback = fallback.strip(' \'"{}')
            data = {"translated": fallback}

        # 9) Context retention
        if self._CONTEXT_RETENTION:
            self.messages.append({'role': 'assistant', 'content': json_str})
        else:
            # remove any placeholder assistant message
            if self.messages and self.messages[-1]['role'] == 'assistant':
                self.messages.pop()

        return data
