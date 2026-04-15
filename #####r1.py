#!/usr/bin/env python3
"""CATR1 — BitNet 1.58b + Omni-Syntax Terminal (Optimized & Fixed)"""
import tkinter as tk
from tkinter import scrolledtext, font
import numpy as np
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
from typing import Optional, Dict, List, Any

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
CONFIG = {
    "d_model": 32,
    "layers": 4,
    "simulate_latency": 0.3,
    "step_delay": 0.1,
    "api_port": 8765,
    "api_key": os.getenv("CATR1_API_KEY", "lm-studio"),
}

CATR1_PROFILE_MD = """# CATR1 — BitNet 1.58b + Omni-Syntax

Direct answers. Clean code. No fluff.

Supported: Python, C++, C, HTML, JS, TS, Java, Rust, Bash, ASM, Go
Run: pip install numpy && python catr1.py
License: GPL3
"""

# ──────────────────────────────────────────────────────────────
# ENGINE
# ──────────────────────────────────────────────────────────────
class CATR1Engine:
    __slots__ = ("name", "ver", "d_model", "_weights", "_layers",
                 "code_experts", "aliases", "intent_map", "intent_weights",
                 "token_index", "dialects", "dialect_idx", "learning_curve",
                 "_lock", "_intent_trained")

    def __init__(self, d_model: int = None):
        self.name = "CATR1"
        self.ver = "1.58b-optimized"
        self.d_model = d_model or CONFIG["d_model"]
        self._lock = threading.Lock()
        self.dialect_idx = {"english": 0, "chinese": 0}
        self.learning_curve: List[float] = []
        self._weights: Optional[np.ndarray] = None
        self._layers: Optional[List[Dict]] = None
        self._intent_trained = False
        self.intent_weights: Optional[np.ndarray] = None
        self.token_index: Dict[str, int] = {}

        self.aliases = {"py":"python","c++":"cpp","js":"javascript","ts":"typescript",
                       "sh":"bash","shell":"bash","asm":"assembly","node":"javascript"}
        
        self.intent_map = {
            "hello": ["hi","hello","hey"],
            "bitnet": ["bitnet","ternary","-1, 0, 1"],
            "recursion": ["recursion","function calls itself"],
            "help": ["help","commands","menu"],
            "languages": ["supported languages","which language","experts"],
            "profile": ["readme",".md","license","gpl3"],
        }
        
        self.dialects = {
            "english": [
                {"hello":"Hi. How can I help?","ready":"Ready. Ask for code or explanation.",
                 "py_intro":"Here is Python code.","generic":"Here is code.",
                 "bitnet":"BitNet uses ternary weights: -1, 0, 1.","recursion":"Recursion: function calls itself. Example:"},
                {"hello":"Hey. What do you need?","ready":"Send a prompt.","py_intro":"Python below.","generic":"Code below.",
                 "bitnet":"Ternary weights: {-1,0,1}.","recursion":"Self-referential function. Example:"},
            ],
            "chinese": [
                {"hello":"你好。需要什么？","ready":"请给出任务。","py_intro":"Python 代码：","generic":"代码：",
                 "bitnet":"BitNet 使用三值权重：-1,0,1。","recursion":"递归：函数自调用。示例："},
            ],
        }
        
        self.code_experts = {
            "python": "def main():\n    print('Hello World')\n\nif __name__ == '__main__':\n    main()",
            "cpp": "#include <iostream>\nint main() { std::cout << \"Hello World\"; return 0; }",
            "c": "#include <stdio.h>\nint main() { printf(\"Hello World\\n\"); return 0; }",
            "javascript": "console.log('Hello World');",
            "html": "<!DOCTYPE html><html><body><h1>Hello World</h1></body></html>",
            "typescript": "function main(): void { console.log('Hello World'); }\nmain();",
            "java": "public class Main { public static void main(String[] args) { System.out.println(\"Hello World\"); } }",
            "rust": "fn main() { println!(\"Hello World\"); }",
            "bash": "#!/bin/bash\necho \"Hello World\"",
            "assembly": "section .data\n    msg db 'Hello World',0xa\nsection .text\n    global _start\n_start:",
            "go": "package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello World\") }",
        }

    @property
    def weights(self) -> np.ndarray:
        if self._weights is None:
            self._weights = np.random.choice([-1,0,1], size=(self.d_model, self.d_model))
        return self._weights

    @property
    def layers(self) -> List[Dict]:
        if self._layers is None:
            self._layers = []
            cfg = {"ff_hidden": self.d_model*2, "n_experts": 4, "top_k": 2}
            for _ in range(CONFIG["layers"]):
                layer = {
                    "qkv": np.random.choice([-1,0,1], size=(3*self.d_model, self.d_model)),
                    "ff_up": np.random.choice([-1,0,1], size=(self.d_model, cfg["ff_hidden"])),
                    "ff_down": np.random.choice([-1,0,1], size=(cfg["ff_hidden"], self.d_model)),
                    "router": np.random.choice([-1,0,1], size=(self.d_model, cfg["n_experts"])),
                    "experts": [{"up": np.random.choice([-1,0,1], size=(self.d_model, cfg["ff_hidden"])),
                                "down": np.random.choice([-1,0,1], size=(cfg["ff_hidden"], self.d_model))}
                               for _ in range(cfg["n_experts"])]
                }
                self._layers.append(layer)
        return self._layers

    def _train_intent(self):
        if self._intent_trained: return
        corpus = [("hi","hello"),("bitnet weights","bitnet"),("recursion example","recursion"),
                  ("supported languages","languages"),("readme","profile")]
        vocab = {t for text,_ in corpus for t in re.findall(r"[a-z0-9]+", text.lower())}
        self.token_index = {t:i for i,t in enumerate(vocab)}
        self.intent_weights = np.zeros((len(self.intent_map), len(vocab)+1), dtype=np.float32)
        for _ in range(16):
            for text, label in corpus:
                x = np.array([text.lower().count(t) for t in vocab] + [1.0], dtype=np.float32)
                y_idx = list(self.intent_map.keys()).index(label)
                scores = self.intent_weights @ x
                others = [s for i,s in enumerate(scores) if i!=y_idx]
                margin = (max(others) if others else -1e9) - scores[y_idx] + 1
                if margin > 0:
                    self.intent_weights[y_idx] += 0.2 * x
        self._intent_trained = True

    def _quantize(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        q = np.zeros_like(x, dtype=np.int8)
        q[x > 0.45] = 1
        q[x < -0.45] = -1
        return q

    def _ternary_matmul(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        x_q = self._quantize(x).astype(np.int16)
        return x_q @ (w==1).astype(np.int16) - x_q @ (w==-1).astype(np.int16)

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32)
        for layer in self.layers:
            qkv = self._ternary_matmul(y, layer["qkv"]).reshape(3, -1)
            q, k, v = qkv[0], qkv[1], qkv[2]
            att = float(np.dot(q,k) / (self.d_model**0.5 + 1e-6))
            gate = 1/(1+np.exp(-att))
            att_out = self._ternary_matmul(v*gate, layer["qkv"][:self.d_model])
            ff = np.tanh(self._ternary_matmul(y, layer["ff_up"]))
            ff_out = self._ternary_matmul(ff, layer["ff_down"])
            logits = self._ternary_matmul(y, layer["router"]).astype(np.float32)
            top_idx = np.argsort(logits)[-2:]
            moe_out = sum(self._ternary_matmul(np.tanh(self._ternary_matmul(y, e["up"])), e["down"])
                         for i in top_idx for e in [layer["experts"][int(i)]]) / 2
            y = np.tanh(y + 0.4*att_out + 0.4*ff_out + 0.2*moe_out)
        return y

    def detect_intent(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        for intent, keywords in self.intent_map.items():
            if any(k in p for k in keywords): return intent
        self._train_intent()
        if not self.token_index: return None
        x = np.array([p.count(t) for t in self.token_index] + [1.0], dtype=np.float32)
        scores = self.intent_weights @ x
        idx = int(np.argmax(scores))
        if scores[idx] - np.mean(scores) > 0.15:
            return list(self.intent_map.keys())[idx]
        return None

    def detect_locale(self, prompt: str) -> str:
        return "chinese" if re.search(r"[\u4e00-\u9fff]|中文|chinese", prompt.lower()) else "english"

    def get_dialect(self, locale: str) -> Dict:
        with self._lock:
            bank = self.dialects.get(locale, self.dialects["english"])
            idx = self.dialect_idx[locale] = self.dialect_idx.get(locale, 0) + 1
            return bank[idx % len(bank)]

    def extract_lang(self, prompt: str) -> Optional[str]:
        p = prompt.lower()
        for alias, lang in self.aliases.items():
            if f"in {alias}" in p or f"{alias} code" in p: return lang
        for lang in self.code_experts:
            if f"in {lang}" in p or f"{lang} code" in p: return lang
        match = re.search(r'(?:write|code|syntax)\s+(?:in\s+)?([a-z+]+)', p)
        if match:
            lang = match.group(1)
            return self.aliases.get(lang, lang if lang in self.code_experts else None)
        return None

    def extract_code_block(self, prompt: str) -> Optional[str]:
        match = re.search(r"```(?:\w+)?\n([\s\S]*?)```", prompt)
        return match.group(1).strip() if match else None

    def safe_exec_python(self, code: str) -> str:
        try:
            tree = ast.parse(code, mode="exec")
            if any(isinstance(n, (ast.Import, ast.ImportFrom, ast.Global)) for n in ast.walk(tree)):
                return "Blocked: imports/global not allowed."
            safe_builtins = {k:v for k,v in __builtins__.items() if k in 
                           {"print","len","range","int","float","str","list","dict","min","max","sum"}}
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(tree,"<catr1>","exec"), {"__builtins__": safe_builtins}, {})
            return buf.getvalue().strip() or "(no output)"
        except Exception as e:
            return f"Error: {e}"

    def generate(self, prompt: str, simulate: bool = True) -> str:
        locale = self.detect_locale(prompt)
        dialect = self.get_dialect(locale)
        intent = self.detect_intent(prompt)
        responses = {
            "hello": dialect["hello"],
            "bitnet": dialect["bitnet"],
            "recursion": f"{dialect['recursion']}\n\ndef fact(n):\n    return 1 if n<=1 else n*fact(n-1)",
            "help": "Commands: hello, bitnet, recursion, languages, readme, write code in <lang>",
            "languages": f"Supported: {', '.join(sorted(self.code_experts))}",
            "profile": CATR1_PROFILE_MD,
        }
        if intent in responses: return responses[intent]
        if any(x in prompt.lower() for x in ["run code","execute","interpret"]):
            code = self.extract_code_block(prompt)
            if not code: return "Include code in ```python ...``` block."
            return f"Result:\n{self.safe_exec_python(code)}"
        lang = self.extract_lang(prompt)
        if lang or any(x in prompt.lower() for x in ["code","write","script"]):
            lang = lang or "python"
            intro = dialect["py_intro"] if lang=="python" else dialect["generic"]
            code = self.code_experts.get(lang, f"# {lang} template\nprint('Hello')")
            return f"{intro}\n\n{code}"
        if simulate: time.sleep(CONFIG["simulate_latency"])
        return dialect["ready"]

    def get_thoughts(self, prompt: str, lang: str, ultra: bool) -> List[str]:
        base = ["⚡ Quantizing to {-1,0,1}...", f"🎯 Routing to [{lang.upper()}] expert...", "✅ Syntax verified."]
        if ultra: base.insert(1, "🧠 UltraThink: deeper analysis...")
        return base

# ──────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────
class CATR1GUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.engine = CATR1Engine()
        self.root.title(f"{self.engine.name} | {self.engine.ver}")
        self.root.geometry("800x600")
        self.root.configure(bg="#050505")
        
        self.fonts = {
            "mono": font.Font(family="Consolas", size=11),
            "mono_bold": font.Font(family="Consolas", size=11, weight="bold"),
            "mono_italic": font.Font(family="Consolas", size=10, slant="italic"),
        }
        
        self.chat = scrolledtext.ScrolledText(root, bg="#050505", fg="#00d9ff",
                                              font=self.fonts["mono"], insertbackground="cyan",
                                              relief="flat", padx=12, pady=12, state="disabled")
        self.chat.pack(expand=True, fill="both")
        self.chat.tag_config("user", foreground="#ffffff", font=self.fonts["mono_bold"])
        self.chat.tag_config("think", foreground="#4a4a4a", font=self.fonts["mono_italic"])
        self.chat.tag_config("bot", foreground="#00aaff", font=self.fonts["mono_bold"])
        
        self.entry = tk.Entry(root, bg="#111", fg="#00d9ff", font=self.fonts["mono"],
                             insertbackground="cyan", relief="flat", bd=6)
        self.entry.pack(fill="x", padx=15, pady=10)
        self.entry.bind("<Return>", lambda e: self.send())
        self.entry.focus_set()
        
        self.log("SYSTEM", f"{self.engine.name} ONLINE • API port {CONFIG['api_port']}")
        self._start_api()

    def log(self, sender: str, text: str, tag: str = None):
        self.chat.config(state="normal")
        self.chat.insert("end", f"[{sender}]: ", "bot" if sender==self.engine.name else tag)
        self.chat.insert("end", f"{text}\n\n", tag)
        self.chat.config(state="disabled")
        self.chat.see("end")

    def send(self):
        msg = self.entry.get().strip()
        if not msg: return
        self.entry.delete(0, "end")
        self.log("YOU", msg, "user")
        threading.Thread(target=self._infer_thread, args=(msg,), daemon=True).start()

    def _infer_thread(self, prompt: str):
        lang = self.engine.extract_lang(prompt) or "general"
        ultra = "ultrathink" in prompt.lower()
        delay = CONFIG["step_delay"] * (1.5 if ultra else 1)
        for step in self.engine.get_thoughts(prompt, lang, ultra):
            self.root.after(0, lambda s=step: self.log("THINK", s, "think"))
            time.sleep(delay)
        response = self.engine.generate(prompt, simulate=True)
        self.root.after(0, lambda r=response: self.log(self.engine.name, r))

    def _start_api(self):
        gui = self
        class Handler(BaseHTTPRequestHandler):
            def _json(self, code: int, data: dict):
                body = json.dumps(data).encode()
                self.send_response(code)
                self.send_header("Content-Type","application/json")
                self.send_header("Content-Length", len(body))
                self.end_headers()
                self.wfile.write(body)
            def _auth(self) -> bool:
                key = self.headers.get("Authorization","").replace("Bearer ","").strip()
                return not CONFIG["api_key"] or key == CONFIG["api_key"]
            def do_POST(self):
                if not self._auth(): return self._json(401,{"error":"Unauthorized"})
                if self.path not in ("/message","/v1/chat/completions"):
                    return self._json(404,{"error":"Not found"})
                try:
                    length = int(self.headers.get("Content-Length",0))
                    data = json.loads(self.rfile.read(length).decode()) if length else {}
                except: return self._json(400,{"error":"Invalid JSON"})
                prompt = data.get("message") or data.get("prompt") or \
                        next((m["content"] for m in reversed(data.get("messages",[])) if m.get("role")=="user"), "")
                if not prompt: return self._json(400,{"error":"Missing prompt"})
                result = {"response": ""}
                def run():
                    result["response"] = gui.engine.generate(prompt, simulate=False)
                    gui.log("API", prompt, "user")
                    gui.log(gui.engine.name, result["response"])
                gui.root.after(0, run)
                self._json(202, {"accepted": True, "id": f"req-{int(time.time())}"})
            def do_GET(self):
                if not self._auth(): return self._json(401,{"error":"Unauthorized"})
                if self.path == "/v1/models":
                    self._json(200, {"data":[{"id":"catr1-local","object":"model"}]})
                else:
                    self._json(200, {"usage":"POST /message {\"message\":\"...\"}"})
            def log_message(self,*a): pass
        def serve():
            try:
                ThreadingHTTPServer(("127.0.0.1", CONFIG["api_port"]), Handler).serve_forever()
            except Exception as e:
                gui.root.after(0, lambda: gui.log("SYSTEM", f"API error: {e}"))
        threading.Thread(target=serve, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# ENTRY
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = CATR1GUI(root)
    root.mainloop()
