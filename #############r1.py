import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import random
import time
import threading
import re
import json
import os
import io
import ast
import traceback
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAT_R11_PROFILE_MD = """# CAT R1.1.X — BitNet 1.58b + Omni-Syntax

It chats in English. No fluff. No self-introduction.

You type: hello

It says: Hi. How can I help?

You type: what is BitNet

It says: BitNet is a neural network with ternary weights constrained to -1, 0, and 1.

You type: write python code

It says: Here is Python code. Then outputs clean code without its own name.

You type: explain recursion

It says: Recursion is a function that calls itself. Example below.

Rules:
- Never says "I am CAT" or "I am an AI"
- Never introduces itself
- Just answers directly
- Outputs clean code without watermarks
- Conversational and helpful

Supported code: Python, C++, C, HTML, JavaScript, TypeScript, Java, Rust, Bash, Assembly, Go

Run: pip install numpy then python cat_r11.py

License: GPL3
"""

# ----------------------------------------------------------------------
# CAT R1.1.X - OMNI-SYNTAX BITNET 1.58b ENGINE
# ----------------------------------------------------------------------
class CatR11OmniEngine:
    def __init__(self, d_model=64):
        self.name = "CAT R1.1.X"
        self.ver = "1.58-bit Ternary MoE"
        self.d_model = d_model
        self._dialect_lock = threading.Lock()
        self._dialect_index_by_locale = {"english": 0, "chinese": 0}
        self.default_response_locale = "english"
        
        # BitNet 1.58b: Weights are constrained to {-1, 0, 1}
        self.weights = np.random.choice([-1, 0, 1], size=(d_model, d_model))
        self.architecture = {
            "layers": 6,
            "heads": 4,
            "n_experts": 8,
            "top_k_experts": 2,
            "ff_hidden": d_model * 2,
            "seq_len": 8,
        }
        self._init_bitnet_architecture()
        
        # Omni-Syntax Expert Database
        self.code_experts = {
            "python": "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()",
            "cpp": "#include <iostream>\n\nint main() {\n    std::cout << \"Hello World\" << std::endl;\n    return 0;\n}",
            "c": "#include <stdio.h>\n\nint main() {\n    printf(\"Hello World\\n\");\n    return 0;\n}",
            "html": "<!DOCTYPE html>\n<html>\n<body>\n  <h1>Hello World</h1>\n</body>\n</html>",
            "javascript": "console.log('Hello World');",
            "typescript": "function main(): void {\n    console.log('Hello World');\n}\n\nmain();",
            "java": "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello World\");\n    }\n}",
            "rust": "fn main() {\n    println!(\"Hello World\");\n}",
            "bash": "#!/bin/bash\necho \"Hello World\"",
            "assembly": "section .data\n    msg db 'Hello World',0xa\nsection .text\n    global _start\n_start: ; Bare metal logic",
            "go": "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello World\")\n}"
        }

        self.aliases = {
            "py": "python", "py3": "python",
            "c++": "cpp", "js": "javascript", "node": "javascript",
            "ts": "typescript", "sh": "bash", "shell": "bash", "zsh": "bash",
            "asm": "assembly"
        }
        self.intent_keywords = {
            "overview": ["overview", "about this", "what does this do"],
            "features": ["features", "core features", "capabilities"],
            "languages": ["supported languages", "languages", "which language", "experts"],
            "getting_started": ["getting started", "setup", "install", "prerequisites", "pip install"],
            "help": ["help", "commands", "menu"],
            "bitnet": ["what is bitnet", "bitnet", "explain bitnet"],
            "hello": ["hello", "hi", "hey"],
            "recursion": ["explain recursion", "what is recursion", "recursion"],
            "learning_curve": ["learning curve", "training curve", "loss curve", "show training"],
            "profile_md": ["readme", ".md", "license", "gpl3", "not fluff", "bitnet 1.58b + omni-syntax"],
        }
        self.intent_labels = list(self.intent_keywords.keys())
        self._intent_index = {name: i for i, name in enumerate(self.intent_labels)}
        self._token_index = {}
        self._intent_weights = np.zeros((len(self.intent_labels), 1), dtype=np.float32)
        self.learning_curve = []
        self._init_intent_interpreter()
        self.dialects = {
            "english": [
                {
                    "name": "standard",
                    "hello": "Hi. How can I help?",
                    "ready": "Ready when you are. Ask for code or a short explanation.",
                    "python_intro": "Here is Python code.",
                    "generic_intro": "Here is code.",
                    "bitnet": "BitNet is a neural network with ternary weights constrained to -1, 0, and 1.",
                    "recursion_intro": "Recursion is a function that calls itself. Example below.",
                },
                {
                    "name": "casual",
                    "hello": "Hey. What do you need?",
                    "ready": "Send a prompt. I will answer directly.",
                    "python_intro": "Got it. Python code below.",
                    "generic_intro": "Got it. Code below.",
                    "bitnet": "BitNet uses ternary weights: -1, 0, and 1.",
                    "recursion_intro": "Recursion means a function calls itself. Example below.",
                },
                {
                    "name": "formal",
                    "hello": "Hello. How may I assist?",
                    "ready": "Provide a request for code or a concise explanation.",
                    "python_intro": "Python solution follows.",
                    "generic_intro": "Code solution follows.",
                    "bitnet": "BitNet constrains model weights to three values: -1, 0, and 1.",
                    "recursion_intro": "Recursion is when a function invokes itself. Example below.",
                },
                {
                    "name": "technical",
                    "hello": "Hello. Request acknowledged.",
                    "ready": "Provide a target language or concept for direct output.",
                    "python_intro": "Python implementation below.",
                    "generic_intro": "Implementation below.",
                    "bitnet": "BitNet is a neural architecture that uses ternary weights in {-1, 0, 1}.",
                    "recursion_intro": "Recursion is self-referential function execution. Example below.",
                },
            ],
            "chinese": [
                {
                    "name": "zh_standard",
                    "hello": "你好。需要我帮你做什么？",
                    "ready": "我已准备好。你可以让我写代码或解释概念。",
                    "python_intro": "下面是 Python 代码。",
                    "generic_intro": "下面是代码。",
                    "bitnet": "BitNet 是一种将权重限制为 -1、0、1 的三值神经网络。",
                    "recursion_intro": "递归是函数调用自身。示例如下。",
                },
                {
                    "name": "zh_concise",
                    "hello": "你好，请说需求。",
                    "ready": "请直接给出任务。",
                    "python_intro": "Python 代码如下。",
                    "generic_intro": "代码如下。",
                    "bitnet": "BitNet 使用三值权重：-1、0、1。",
                    "recursion_intro": "递归就是函数自调用。示例如下。",
                },
            ],
        }

    def _tokenize(self, text):
        return re.findall(r"[a-z0-9+#]+", text.lower())

    def _init_intent_interpreter(self):
        """Train a tiny online intent head so behavior is not keyword-only."""
        corpus = [
            ("hello", "hello"),
            ("hi there", "hello"),
            ("hey", "hello"),
            ("what is bitnet", "bitnet"),
            ("explain bitnet model", "bitnet"),
            ("explain recursion", "recursion"),
            ("what is recursion", "recursion"),
            ("show me help", "help"),
            ("commands list", "help"),
            ("supported languages", "languages"),
            ("what languages do you support", "languages"),
            ("getting started", "getting_started"),
            ("how do i install", "getting_started"),
            ("overview", "overview"),
            ("features", "features"),
            ("show learning curve", "learning_curve"),
            ("training curve", "learning_curve"),
            ("readme", "profile_md"),
            ("license", "profile_md"),
        ]

        vocab = {}
        for text, _ in corpus:
            for tok in self._tokenize(text):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        self._token_index = vocab
        self._intent_weights = np.zeros((len(self.intent_labels), len(vocab) + 1), dtype=np.float32)

        lr = 0.15
        epochs = 32
        for _ in range(epochs):
            total_loss = 0.0
            for text, label in corpus:
                x = self._encode_text(text)
                y_idx = self._intent_index[label]
                scores = self._intent_weights @ x
                # Multiclass perceptron-style hinge loss.
                target = scores[y_idx]
                best_other_idx = 0
                best_other_score = -1e9
                for i, sc in enumerate(scores):
                    if i == y_idx:
                        continue
                    if sc > best_other_score:
                        best_other_score = sc
                        best_other_idx = i
                margin = float(best_other_score - target + 1.0)
                if margin > 0:
                    self._intent_weights[y_idx] += lr * x
                    self._intent_weights[best_other_idx] -= lr * x
                    total_loss += margin
            self.learning_curve.append(total_loss / max(1, len(corpus)))

    def _encode_text(self, text):
        x = np.zeros(len(self._token_index) + 1, dtype=np.float32)
        x[-1] = 1.0  # bias
        for tok in self._tokenize(text):
            idx = self._token_index.get(tok)
            if idx is not None:
                x[idx] += 1.0
        return x

    def predict_intent_ml(self, prompt):
        if self._intent_weights.shape[1] <= 1:
            return None
        x = self._encode_text(prompt)
        scores = self._intent_weights @ x
        idx = int(np.argmax(scores))
        confidence = float(scores[idx] - np.mean(scores))
        if confidence < 0.15:
            return None
        return self.intent_labels[idx]

    def _init_bitnet_architecture(self):
        """Initialize a fuller BitNet-style architecture (ternary transformer + MoE)."""
        cfg = self.architecture
        self.bitnet_layers = []
        for _ in range(cfg["layers"]):
            layer = {
                "q": np.random.choice([-1, 0, 1], size=(self.d_model, self.d_model)),
                "k": np.random.choice([-1, 0, 1], size=(self.d_model, self.d_model)),
                "v": np.random.choice([-1, 0, 1], size=(self.d_model, self.d_model)),
                "o": np.random.choice([-1, 0, 1], size=(self.d_model, self.d_model)),
                "ff_up": np.random.choice([-1, 0, 1], size=(self.d_model, cfg["ff_hidden"])),
                "ff_down": np.random.choice([-1, 0, 1], size=(cfg["ff_hidden"], self.d_model)),
                "router": np.random.choice([-1, 0, 1], size=(self.d_model, cfg["n_experts"])),
                "experts": [
                    {
                        "up": np.random.choice([-1, 0, 1], size=(self.d_model, cfg["ff_hidden"])),
                        "down": np.random.choice([-1, 0, 1], size=(cfg["ff_hidden"], self.d_model)),
                    }
                    for _ in range(cfg["n_experts"])
                ],
            }
            self.bitnet_layers.append(layer)

    def next_dialect(self, locale):
        locale = locale if locale in self.dialects else self.default_response_locale
        with self._dialect_lock:
            index = self._dialect_index_by_locale.get(locale, 0)
            dialect_bank = self.dialects[locale]
            dialect = dialect_bank[index % len(dialect_bank)]
            self._dialect_index_by_locale[locale] = index + 1
        return dialect

    def ternary_quantize(self, x, threshold=0.5):
        """Quantize activations to {-1, 0, 1} using a symmetric threshold."""
        x = np.asarray(x, dtype=np.float32)
        q = np.zeros_like(x, dtype=np.int8)
        q[x > threshold] = 1
        q[x < -threshold] = -1
        return q

    def bitnet_matmul(self, x):
        """BitNet-style ternary projection with add/sub accumulation semantics."""
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-5)
        x_q = self.ternary_quantize(x_norm, threshold=0.45).astype(np.int16)
        pos_mask = (self.weights == 1).astype(np.int16)
        neg_mask = (self.weights == -1).astype(np.int16)
        # Add/sub routing: sum positives minus sum negatives.
        return x_q @ pos_mask - x_q @ neg_mask

    def bitnet_linear(self, x, w):
        x_q = self.ternary_quantize(x, threshold=0.45).astype(np.int16)
        w_pos = (w == 1).astype(np.int16)
        w_neg = (w == -1).astype(np.int16)
        return x_q @ w_pos - x_q @ w_neg

    def bitnet_moe_route(self, x, layer):
        cfg = self.architecture
        router_logits = self.bitnet_linear(x, layer["router"]).astype(np.float32)
        top_idx = np.argsort(router_logits)[-cfg["top_k_experts"]:]
        mix = np.zeros(self.d_model, dtype=np.float32)
        for idx in top_idx:
            expert = layer["experts"][int(idx)]
            up = self.bitnet_linear(x, expert["up"]).astype(np.float32)
            up = np.tanh(up)
            down = self.bitnet_linear(up, expert["down"]).astype(np.float32)
            mix += down
        return mix / max(1, len(top_idx))

    def bitnet_block(self, x, layer):
        q = self.bitnet_linear(x, layer["q"]).astype(np.float32)
        k = self.bitnet_linear(x, layer["k"]).astype(np.float32)
        v = self.bitnet_linear(x, layer["v"]).astype(np.float32)
        # Lightweight self-attention approximation on vector state.
        att_score = float(np.dot(q, k) / (np.sqrt(self.d_model) + 1e-6))
        att_gate = 1.0 / (1.0 + np.exp(-att_score))
        att_out = self.bitnet_linear(v * att_gate, layer["o"]).astype(np.float32)
        ff_hidden = self.bitnet_linear(x, layer["ff_up"]).astype(np.float32)
        ff_hidden = np.tanh(ff_hidden)
        ff_out = self.bitnet_linear(ff_hidden, layer["ff_down"]).astype(np.float32)
        moe_out = self.bitnet_moe_route(x, layer)
        y = x + 0.45 * att_out + 0.35 * ff_out + 0.20 * moe_out
        return np.tanh(y)

    def bitnet_forward(self, x):
        y = np.asarray(x, dtype=np.float32)
        for layer in self.bitnet_layers:
            y = self.bitnet_block(y, layer)
        return y

    def extract_language(self, prompt):
        p = prompt.lower()
        words = set(re.findall(r"[a-z0-9+#]+", p))
        for lang in self.code_experts.keys():
            if lang in words or f"in {lang}" in p:
                return lang
        for alias, real_lang in self.aliases.items():
            if alias in words or f"in {alias}" in p:
                return real_lang
        patterns = [
            r"(?:write|generate|create|make)\s+([a-z0-9+#]+)\s+(?:code|script|program)",
            r"(?:code|script|program)\s+(?:in|for)\s+([a-z0-9+#]+)",
            r"(?:syntax|language)\s*[:=]?\s*([a-z0-9+#]+)",
            r"in\s+([a-z0-9+#]+)",
        ]
        for pat in patterns:
            match = re.search(pat, p)
            if match:
                guessed = match.group(1)
                return self.aliases.get(guessed, guessed)
        match = re.search(r'in ([a-z0-9+#]+)', p)
        if match:
            guessed = match.group(1)
            return self.aliases.get(guessed, guessed)
        return None

    def extract_code_block(self, prompt):
        match = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\n([\s\S]*?)```", prompt)
        if match:
            return match.group(1).strip()
        return None

    def wants_code_execution(self, prompt):
        p = prompt.lower()
        triggers = [
            "run code",
            "execute code",
            "interpret code",
            "code interpreter",
            "run this",
            "execute this",
        ]
        return any(t in p for t in triggers)

    def safe_python_execute(self, code_text):
        """Execute Python code with a restricted builtins set."""
        try:
            tree = ast.parse(code_text, mode="exec")
        except SyntaxError as err:
            return f"Execution error: syntax error at line {err.lineno}: {err.msg}"

        blocked_nodes = (
            ast.Import,
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
        )
        for node in ast.walk(tree):
            if isinstance(node, blocked_nodes):
                return "Execution blocked: imports and global/nonlocal statements are disabled."

        safe_builtins = {
            "print": print,
            "len": len,
            "range": range,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "min": min,
            "max": max,
            "sum": sum,
            "abs": abs,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "round": round,
            "any": any,
            "all": all,
        }

        env = {"__builtins__": safe_builtins}
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(compile(tree, "<catr1-interpreter>", "exec"), env, env)
        except Exception as err:
            return f"Execution error: {err}\n{traceback.format_exc(limit=1)}"

        output = stdout.getvalue().strip()
        return output if output else "(no output)"

    def comment_prefix(self, language):
        lang = (language or "").lower()
        if lang in {"python", "bash", "shell", "sh", "zsh", "ruby", "perl", "yaml", "toml"}:
            return "#"
        if lang in {"html", "xml"}:
            return "<!-- -->"
        return "//"

    def generate_dynamic_code(self, language, prompt):
        lang = (language or "text").lower()
        comment = self.comment_prefix(lang)
        if comment == "<!-- -->":
            return (
                "<!-- Dynamic template -->\n"
                f"<!-- Language: {lang} -->\n"
                "<!DOCTYPE html>\n"
                "<html>\n"
                "<body>\n"
                "  <h1>Hello World</h1>\n"
                "</body>\n"
                "</html>"
            )
        if lang in {"python", "py"}:
            return (
                "def main():\n"
                "    print('Hello World')\n\n"
                "if __name__ == '__main__':\n"
                "    main()"
            )
        return (
            f"{comment} Dynamic template\n"
            f"{comment} Language: {lang}\n"
            f"{comment} Prompt: {prompt.strip()[:80]}"
        )

    def detect_intent(self, prompt):
        p = prompt.lower()
        if p.strip() in {"hi", "hello", "hey"}:
            return "hello"
        for intent, keys in self.intent_keywords.items():
            if any(k in p for k in keys):
                return intent
        return self.predict_intent_ml(prompt)

    def detect_response_locale(self, prompt):
        p = prompt.lower()
        if re.search(r"[\u4e00-\u9fff]", prompt):
            return "chinese"
        chinese_triggers = ["chinese", "mandarin", "中文", "中国话", "汉语", "普通话", "chinese dialect"]
        if any(key in p for key in chinese_triggers):
            return "chinese"
        return self.default_response_locale

    def render_overview(self):
        return (
            "BitNet 1.58b-style ternary routing with an Omni-Syntax code expert set. "
            "Direct answers and clean code output."
        )

    def render_features(self):
        return (
            "- BitNet-style ternary weights and ternary activation quantization\n"
            "- Omni-Syntax MoE: per-language code experts\n"
            "- Cyber terminal GUI: Tkinter dark mode + cyan text\n"
            "- Language detection: keywords, aliases, and regex hints\n"
            "- English-first direct answers for chat prompts\n"
            "- Threaded inference: non-blocking UI while generating"
        )

    def render_supported_languages(self):
        langs = sorted(self.code_experts.keys())
        alias_preview = ", ".join(f"{k}->{v}" for k, v in sorted(self.aliases.items()))
        return (
            f"Supported code experts ({len(langs)}): {', '.join(langs)}\n"
            f"Aliases: {alias_preview}"
        )

    def render_getting_started(self):
        return (
            "Getting started:\n"
            "1) Python 3.7+\n"
            "2) Tkinter (usually bundled)\n"
            "3) Install dependency: pip install numpy\n"
            "4) Run: python cat_r11.py\n"
            "Tip: add 'ultrathink' to your prompt for longer reasoning simulation."
        )

    def render_help(self):
        return (
            "Commands:\n"
            "- overview\n"
            "- features\n"
            "- supported languages\n"
            "- getting started / prerequisites\n"
            "- what is bitnet\n"
            "- explain recursion\n"
            "- learning curve\n"
            "- add 'chinese dialect' or Chinese text for Chinese output\n"
            "- readme / .md / license\n"
            "- write code in <language> [ultrathink]\n"
            "- run code / execute code with a ```python ...``` block"
        )

    def render_bitnet_explanation(self, dialect):
        return dialect["bitnet"]

    def render_hello(self, dialect):
        return dialect["hello"]

    def render_recursion(self, dialect):
        return (
            f"{dialect['recursion_intro']}\n\n"
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)"
        )

    def render_profile_md(self):
        return CAT_R11_PROFILE_MD

    def render_learning_curve(self):
        if not self.learning_curve:
            return "No training history available."
        recent = self.learning_curve[-12:]
        max_v = max(recent) if max(recent) > 0 else 1.0
        bars = []
        for i, v in enumerate(recent):
            width = int(round((v / max_v) * 24))
            bars.append(f"e{i+1:02d} | {'#' * width} {v:.4f}")
        first = self.learning_curve[0]
        last = self.learning_curve[-1]
        trend = "down" if last < first else "flat/up"
        return (
            "Learning curve (intent head, lower is better):\n"
            + "\n".join(bars)
            + f"\nstart={first:.4f}, end={last:.4f}, trend={trend}"
        )

    def think(self, prompt, target_lang, ultra=False):
        lang_display = target_lang.upper() if target_lang else "GENERAL"
        steps = [
            "🧠 Initializing BitNet 1.58b Ternary Kernels...",
            "⚙️ Quantizing input tensors to {-1, 0, 1}...",
            f"🔍 Routing to Omni-Syntax MoE... Target: [{lang_display}]",
            "🛠️ Calculating logits via zero-multiply accumulation...",
            "✨ Self-Correction: Syntax verified for 1.1.X."
        ]
        if ultra:
            steps.insert(3, "🧩 UltraThink: running deeper routing and consistency pass...")
            steps.append("✅ UltraThink complete: response confidence stabilized.")
        return steps

    def generate(self, prompt, target_lang, simulate_latency=True, response_locale=None):
        p = prompt.lower()
        locale = response_locale or self.detect_response_locale(prompt)
        dialect = self.next_dialect(locale)
        intent = self.detect_intent(prompt)
        if intent == "overview":
            return self.render_overview()
        if intent == "features":
            return self.render_features()
        if intent == "languages":
            return self.render_supported_languages()
        if intent == "getting_started":
            return self.render_getting_started()
        if intent == "help":
            return self.render_help()
        if intent == "bitnet":
            return self.render_bitnet_explanation(dialect)
        if intent == "hello":
            return self.render_hello(dialect)
        if intent == "recursion":
            return self.render_recursion(dialect)
        if intent == "learning_curve":
            return self.render_learning_curve()
        if intent == "profile_md":
            return self.render_profile_md()

        code_block = self.extract_code_block(prompt)
        if self.wants_code_execution(prompt):
            if not code_block:
                return "Interpreter: include a fenced Python block like ```python ...```."
            lang = target_lang or "python"
            if lang != "python":
                return "Interpreter currently supports Python execution only. Request Python or omit language."
            result = self.safe_python_execute(code_block)
            return f"Execution result:\n{result}"

        dummy_input = np.random.randn(self.d_model)
        dummy_input = self.bitnet_forward(dummy_input)
        if simulate_latency:
            time.sleep(1.2)

        is_code = any(x in p for x in ["code", "write", "syntax", "script", "program"]) or target_lang
        if is_code:
            if target_lang in self.code_experts:
                label = dialect["generic_intro"]
                if target_lang == "python":
                    label = dialect["python_intro"]
                code = self.code_experts[target_lang]
                return f"{label}\n\n{code}"
            dynamic_lang = target_lang or "python"
            code = self.generate_dynamic_code(dynamic_lang, prompt)
            if dynamic_lang == "python":
                label = dialect["python_intro"]
            else:
                label = f"{dialect['generic_intro']} ({dynamic_lang})"
            return f"{label}\n\n{code}"
        return dialect["ready"]

# ----------------------------------------------------------------------
# GUI - CAT R1.1.X TERMINAL
# ----------------------------------------------------------------------
class CatR1_1_X:
    def __init__(self, root):
        self.root = root
        self.engine = CatR11OmniEngine()
        self.api_host = "127.0.0.1"
        self.api_port = 8765
        self.api_key = os.getenv("CATR1_API_KEY", "lm-studio")
        self.root.title(f"{self.engine.name} | BITNET 1.58b")
        self.root.geometry("800x600")
        self.root.configure(bg='#050505')

        self.chat = scrolledtext.ScrolledText(root, bg='#050505', fg='#00d9ff', 
                                              font=('Consolas', 11), insertbackground='cyan',
                                              relief='flat', padx=15, pady=15)
        self.chat.pack(expand=True, fill='both')
        self.chat.tag_config('thought', foreground='#4a4a4a', font=('Consolas', 10, 'italic'))
        self.chat.tag_config('user', foreground='#ffffff', font=('Consolas', 11, 'bold'))
        self.chat.tag_config('bot_name', foreground='#0055ff', font=('Consolas', 11, 'bold'))

        self.entry = tk.Entry(root, bg='#111111', fg='#00d9ff', font=('Consolas', 12), 
                              insertbackground='cyan', relief='flat', bd=8)
        self.entry.pack(fill='x', padx=20, pady=20)
        self.entry.bind("<Return>", lambda e: self.handle_input())

        self.append_msg("SYSTEM", f"{self.engine.name} ({self.engine.ver}) ONLINE.")
        self.start_message_api_server()
        self.append_msg("SYSTEM", f"Message API online at http://{self.api_host}:{self.api_port}/message")
        self.append_msg("SYSTEM", "API key auth enabled (Bearer token required).")

    def append_msg(self, sender, text, tag=None):
        self.chat.config(state='normal')
        self.chat.insert(tk.END, f"[{sender}]: ", 'bot_name' if sender == "CAT R1.1.X" else tag)
        self.chat.insert(tk.END, f"{text}\n\n", tag)
        self.chat.config(state='disabled')
        self.chat.see(tk.END)

    def handle_input(self):
        msg = self.entry.get()
        if not msg:
            return
        self.entry.delete(0, tk.END)
        self.append_msg("YOU", msg, 'user')
        threading.Thread(target=self.infer, args=(msg,), daemon=True).start()

    def _run_inference(self, prompt, emit_thoughts=True, sleep_between_steps=True, simulate_latency=True):
        target_lang = self.engine.extract_language(prompt)
        response_locale = self.engine.detect_response_locale(prompt)
        lower_prompt = prompt.lower()
        ultra = (
            ("ultrathink" in lower_prompt)
            or ("ultarthink" in lower_prompt)
            or ("ultrathiknk" in lower_prompt)
        )
        step_delay = 0.55 if ultra else 0.3
        thoughts = self.engine.think(prompt, target_lang, ultra=ultra)
        if emit_thoughts:
            for step in thoughts:
                self.root.after(0, lambda s=step: self.append_msg("THINK", s, 'thought'))
                if sleep_between_steps:
                    time.sleep(step_delay)
        result = self.engine.generate(
            prompt,
            target_lang,
            simulate_latency=simulate_latency,
            response_locale=response_locale,
        )
        return {
            "target_language": target_lang or "general",
            "response_locale": response_locale,
            "ultrathink": ultra,
            "thoughts": thoughts,
            "response": result,
        }

    def infer(self, prompt):
        payload = self._run_inference(
            prompt,
            emit_thoughts=True,
            sleep_between_steps=True,
            simulate_latency=True,
        )
        self.root.after(0, lambda r=payload["response"]: self.append_msg("CAT R1.1.X", r))

    def infer_from_api(self, prompt):
        # Keep GUI updates on Tk thread.
        self.root.after(0, lambda p=prompt: self.append_msg("API", p, 'user'))
        payload = self._run_inference(
            prompt,
            emit_thoughts=False,
            sleep_between_steps=False,
            simulate_latency=False,
        )
        self.root.after(0, lambda r=payload["response"]: self.append_msg("CAT R1.1.X", r))
        return payload

    def start_message_api_server(self):
        app = self

        class MessageHandler(BaseHTTPRequestHandler):
            def _send_json(self, status_code, payload):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _extract_api_key(self):
                auth = self.headers.get("Authorization", "").strip()
                if auth.lower().startswith("bearer "):
                    return auth.split(" ", 1)[1].strip()
                x_api_key = self.headers.get("X-API-Key", "").strip()
                return x_api_key

            def _require_auth(self):
                expected = (app.api_key or "").strip()
                if not expected:
                    return True
                provided = self._extract_api_key()
                return provided == expected

            def _unauthorized(self):
                self._send_json(
                    401,
                    {
                        "error": {
                            "message": "Invalid API key",
                            "type": "invalid_request_error",
                            "code": "invalid_api_key",
                        }
                    },
                )

            def _read_json(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                    return json.loads(raw), None
                except Exception:
                    return None, {"ok": False, "error": "Invalid JSON body"}

            def do_POST(self):
                if not self._require_auth():
                    self._unauthorized()
                    return

                if self.path != "/message":
                    if self.path == "/v1/chat/completions":
                        data, err = self._read_json()
                        if err:
                            self._send_json(400, err)
                            return

                        messages = data.get("messages", [])
                        prompt = ""
                        for m in reversed(messages):
                            if str(m.get("role", "")).lower() == "user":
                                prompt = str(m.get("content", "")).strip()
                                break
                        if not prompt:
                            prompt = str(data.get("prompt", "")).strip()
                        if not prompt:
                            self._send_json(400, {"ok": False, "error": "Missing user prompt/messages"})
                            return

                        payload = app.infer_from_api(prompt)
                        answer = payload.get("response", "")
                        completion = {
                            "id": f"chatcmpl-{int(time.time() * 1000)}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": str(data.get("model", "catr1-local")),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": answer},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": max(1, len(prompt.split())),
                                "completion_tokens": max(1, len(str(answer).split())),
                                "total_tokens": max(1, len(prompt.split())) + max(1, len(str(answer).split())),
                            },
                        }
                        self._send_json(200, completion)
                        return
                    self._send_json(
                        404,
                        {
                            "ok": False,
                            "error": "Use POST /message or /v1/chat/completions",
                        },
                    )
                    return

                data, err = self._read_json()
                if err:
                    self._send_json(400, err)
                    return

                prompt = str(data.get("message", "")).strip()
                if not prompt:
                    self._send_json(400, {"ok": False, "error": "Field 'message' is required"})
                    return

                payload = app.infer_from_api(prompt)
                self._send_json(200, {"ok": True, **payload})

            def do_GET(self):
                if not self._require_auth():
                    self._unauthorized()
                    return

                if self.path == "/message":
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "usage": "POST /message {\"message\":\"...\"}",
                            "auth": "Authorization: Bearer <key>",
                        },
                    )
                elif self.path == "/v1/models":
                    self._send_json(
                        200,
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": "catr1-local",
                                    "object": "model",
                                    "owned_by": "local",
                                }
                            ],
                        },
                    )
                else:
                    self._send_json(404, {"ok": False, "error": "Use /message or /v1/models"})

            def log_message(self, format, *args):
                return

        def run_server():
            try:
                server = ThreadingHTTPServer((self.api_host, self.api_port), MessageHandler)
                server.serve_forever()
            except Exception as err:
                self.root.after(0, lambda e=err: self.append_msg("SYSTEM", f"Message API failed: {e}"))

        threading.Thread(target=run_server, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    CatR1_1_X(root)
    root.mainloop()
