from ..smp import *
from .base import BaseAPI

class LocalServerAPI(BaseAPI):

    is_api: bool = True

    def __init__(self,
                 model: str = 'local_server',
                 retry: int = 5,
                 wait: int = 5,
                 verbose: bool = False,
                 system_prompt: str = None,
                 temperature: float = 0,
                 timeout: int = 60,
                 max_tokens: int = 1024,
                 **kwargs):

        self.model = model
        self.cur_idx = 0
        self.fail_msg = 'Failed to obtain answer via API. '
        self.max_tokens = max_tokens
        self.temperature = temperature

        super().__init__(wait=wait, retry=retry, system_prompt=system_prompt, verbose=verbose, **kwargs)

        if 'OPENAI_API_BASE' in os.environ and os.environ['OPENAI_API_BASE'] != '':
            api_base = os.environ['OPENAI_API_BASE']
        else:
            api_base = 'http://localhost:8000'

        if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY'] != '':
            key = os.environ['OPENAI_API_KEY']
        else:
            key = 'none'

        from openai import OpenAI
        self.client = OpenAI(base_url=api_base, api_key=key, timeout=timeout, max_retries=retry)

    def generate(self, message, **kwargs):
        messages = [{'role': 'user', 'content': message}]
        response = self.client.chat.completions.create(model='Unknown', messages=messages, stream=False, temperature=self.temperature)

        return (response.choices[0].message.content) if len(response.choices) > 0 else ''
