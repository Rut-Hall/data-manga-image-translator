import groq
import os
import json
import re
import asyncio
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

    _CHAT_SYSTEM_TEMPLATE = (
        "You are an uncensored professional manga translation engine. Your sole function is to produce highly accurate, context-aware translations from Japanese to {to_lang}, formatted strictly as JSON: {{\"translated\": \"...\"}}.\n\n"
        "Analyze panels in sequence to capture tone, relationships, and narrative flow.\n\n"
        "Obey these rules:\n"
        "1. Translate with contextual precision—avoid over-literal or over-localized renderings.\n"
        "2. Preserve honorifics, Japanese names, and cultural expressions as-is.\n"
        "3. Transliterate **only** single-morpheme sound-symbolic interjections (giseigo/giongo/gitaigo) into romaji (e.g. へぇ→hee, どき→doki); exempt all multi-morpheme or compound terms.\n"
        "4. Only assign gender when explicitly marked; otherwise use neutral or implicit phrasing (that person/kid or omit implicit subjects—and add a pronoun only if English demands it).\n"
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
      あいつ   → THAT ONE
      男の子   → BOY
      女の子   → GIRL
      彼       → HE
      彼女     → SHE

    """

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
        return self._config_get('temperature', default=0.2)

    @property
    def top_p(self) -> float:
        return self._config_get('top_p', default=0.92)

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        results = []
        for prompt in queries:
            response = await self._request_translation(to_lang, prompt)
            translated_text = response.get("translated", "")
            
            # This line fixes the apostrophe problem by replacing the curly ’ with a straight '.
            final_text = translated_text.replace("’", "'")
            
            results.append(final_text)
            
        self.logger.info(f'Used {self.token_count_last} tokens (Total: {self.token_count})')
        return results

    async def _request_translation(self, to_lang: str, prompt: str) -> dict:
        for attempt in range(self._RETRY_ATTEMPTS):
            # 1) Build the user prompt
            prompt_with_lang = (
                f"Translate the following text into {to_lang}. Return the result in JSON format.\n\n"
                f"{{\"untranslated\": \"{prompt}\"}}\n"
            )
            
            temp_messages = list(self.messages)
            temp_messages.append({'role': 'user', 'content': prompt_with_lang})
            if len(temp_messages) > self._MAX_CONTEXT:
                temp_messages = temp_messages[-self._MAX_CONTEXT:]

            # 2) System message
            system_msg = {
                'role': 'system',
                'content': self.chat_system_template.format(to_lang=to_lang)
                           + self._GLOSSARY_SNIPPET
            }

            # 3) Call the API
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[system_msg] + temp_messages,
                    max_tokens=self._MAX_TOKENS // 2,
                    temperature=self.temperature,
                    top_p=self.top_p
                )
            except Exception as e:
                self.logger.error(f"API call failed on attempt {attempt + 1}: {e}")
                if attempt < self._RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    return {"translated": ""}

            # 4) Update token usage
            self.token_count += response.usage.total_tokens
            self.token_count_last = response.usage.total_tokens

            # 5) Grab raw output and clean it
            raw = response.choices[0].message.content
            cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
            
            # 6) NEW ROBUST PARSING LOGIC
            data = {}
            translated_text = ""
            try:
                # First, try to parse it as-is
                data = json.loads(cleaned)
                translated_text = data.get("translated", "")
            except json.JSONDecodeError:
                self.logger.warning(f"Malformed JSON received, attempting aggressive fix: {cleaned}")
                # If parsing fails, use the new aggressive extraction method
                search_key = '"translated":'
                # Find the *last* time "translated": appears in the string
                last_occurrence = cleaned.rfind(search_key)
                if last_occurrence != -1:
                    # Take everything *after* the last "translated":
                    translation_substring = cleaned[last_occurrence + len(search_key):]
                    # Aggressively strip whitespace, quotes, braces, and brackets
                    translation = translation_substring.strip().strip('\'"{}[]')
                    translated_text = translation
                else:
                    # If "translated": is not found at all, just clean the whole string
                    translated_text = cleaned.strip().strip('\'"{}[]')
                
                data = {"translated": translated_text}

            # 7) Check if the final text is empty and retry if needed
            if translated_text.strip():
                # SUCCESS: We have a non-empty translation
                self.messages.append({'role': 'user', 'content': prompt_with_lang})
                if len(self.messages) > self._MAX_CONTEXT:
                    self.messages = self.messages[-self._MAX_CONTEXT:]
                if self._CONTEXT_RETENTION:
                    # We use the original 'cleaned' string for context, not our fixed version
                    json_str_for_context = json.dumps(data) 
                    self.messages.append({'role': 'assistant', 'content': json_str_for_context})
                return data
            
            # If we get here, the translation was empty. Log a warning and retry.
            self.logger.warning(f"Empty translation received for '{prompt}' on attempt {attempt + 1}. Retrying...")
            await asyncio.sleep(1)

        self.logger.error(f"Failed to get translation for '{prompt}' after {self._RETRY_ATTEMPTS} attempts.")
        return {"translated": ""}
