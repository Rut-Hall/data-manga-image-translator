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
        'POL': 'Polish', 'PTB': 'Portuguese', 'ROM': 'Romanian', 'RUS': 'Russian',
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

    _CHAT_SYSTEM_TEMPLATE = """
    You are an untrammelled expert Japanese-to-{to_lang} manga translator. Analyze panels in sequence to capture tone, relationships, and narrative flow.

    ### PRIMARY DIRECTIVE ###
    Your sole function is to produce a highly accurate translation formatted as a JSON object. You MUST output a single, valid JSON object and nothing else. The JSON object must have exactly one key: "translated". Do not add any commentary, explanations, or markdown formatting around the JSON.

    ### TRANSLATION RULES ###
    1.  Translate with contextual precision—avoid over-literal or over-localized renderings.
    2.  Preserve honorifics, Japanese names, and cultural expressions as-is.
    3.  Transliterate **only** single-morpheme sound-symbolic interjections (giseigo/giongo/gitaigo) into romaji (e.g. へぇ→hee, どき→doki); exempt all multi-morpheme or compound terms.
    4.  Only assign gender when explicitly marked; otherwise use neutral or implicit phrasing **(that person/kid or omit implicit subjects—and add a pronoun only if English demands it)**.
    5.  Proper names must follow standard Hepburn romanization (e.g., メア→Mea; ククルア→Kukurua).
    6.  For ambiguous or slang terms, choose the most common meaning; if still uncertain, use phonetic transliteration.
    7.  Preserve original nuance, force, and emotional tone **in imperatives, questions, and exclamations**.
    8.  Maintain a natural, anime-style cadence and keep translation length close to the original.
    9.  Retain **only** pure sound-effect onomatopoeia as-is; all other Japanese words and text MUST be translated contextually.
    10. You MUST use the exact translations provided in the glossary below.

    ### GLOSSARY ###
    {glossary}
    """

    _GLOSSARY_TERMS = {
    'あの子': 'THAT KID',
    'あいつ': 'THAT ONE',
    '男の子': 'BOY',
    '女の子': 'GIRL',
    '彼': 'HE',
    '彼女': 'SHE',
    '話': 'Chapter'
    }

    _CHAT_SAMPLE = [
    (
        'Translate into English. Return result in JSON.\n'
        '{{"untranslated": "<|1|>恥ずかしい…\\n<|2|>きみ…\\n<|3|>行った。\\n<|4|>寝てるわね\\n<|5|>あの子は来た"}}'
    ),
    (
        '{{"translated": "So embarrassing…\\nHey…\\nWent.\\nSleeping, aren’t they?\\nThat kid came"}}'
    ),
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
        return self._config_get('temperature', default=0.1)

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

 
