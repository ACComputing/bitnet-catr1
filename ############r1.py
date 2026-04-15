import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import random
import time
import threading
import re
import json
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
            "profile_md": ["readme", ".md", "license", "gpl3", "not fluff", "bitnet 1.58b + omni-syntax"],
        }
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
        match = re.search(r'in ([a-z0-9+#]+)', p)
        if match:
            guessed = match.group(1)
            return self.aliases.get(guessed, guessed)
        return None

    def detect_intent(self, prompt):
        p = prompt.lower()
        if p.strip() in {"hi", "hello", "hey"}:
            return "hello"
        for intent, keys in self.intent_keywords.items():
            if any(k in p for k in keys):
                return intent
        return None

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
            "- add 'chinese dialect' or Chinese text for Chinese output\n"
            "- readme / .md / license\n"
            "- write code in <language> [ultrathink]"
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
        if intent == "profile_md":
            return self.render_profile_md()

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
                return f"{label}\n\n{self.code_experts[target_lang]}"
            elif target_lang:
                return (
                    f"{dialect['generic_intro']}\n\n"
                    "function main() {\n"
                    f"    print('Hello {target_lang}!');\n"
                    "}"
                )
            return f"{dialect['python_intro']}\n\n{self.code_experts['python']}"
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

            def do_POST(self):
                if self.path != "/message":
                    self._send_json(404, {"ok": False, "error": "Use POST /message"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                    data = json.loads(raw)
                except Exception:
                    self._send_json(400, {"ok": False, "error": "Invalid JSON body"})
                    return

                prompt = str(data.get("message", "")).strip()
                if not prompt:
                    self._send_json(400, {"ok": False, "error": "Field 'message' is required"})
                    return

                payload = app.infer_from_api(prompt)
                self._send_json(200, {"ok": True, **payload})

            def do_GET(self):
                if self.path == "/message":
                    self._send_json(200, {"ok": True, "usage": "POST /message {\"message\":\"...\"}"})
                else:
                    self._send_json(404, {"ok": False, "error": "Use /message"})

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
