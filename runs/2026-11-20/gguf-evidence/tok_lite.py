import datetime
import json

import jinja2
import jinja2.ext
import jinja2.sandbox
from tokenizers import Tokenizer


class ChatTokenizer:
    """The checkpoint's tokenizer.json and chat_template.jinja, rendered the way transformers renders
    them (a sandboxed Jinja environment with trim_blocks, lstrip_blocks and loop controls)."""

    def __init__(self, model_dir):
        self.tk = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        tok = lambda t: t["content"] if isinstance(t, dict) else t  # noqa: E731
        self.bos_token, self.eos_token = tok(cfg.get("bos_token")), tok(cfg.get("eos_token"))
        self.bos_token_id = self.tk.token_to_id(self.bos_token)
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
        )

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
        self.template = env.from_string((model_dir / "chat_template.jinja").read_text())

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        text = self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            **kwargs,
        )
        return self.encode(text) if tokenize else text

    def encode(self, text, add_special_tokens=False):
        return self.tk.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, token):
        return self.tk.token_to_id(token)
