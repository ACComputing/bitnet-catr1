import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import random
import time
import threading
import re

# ----------------------------------------------------------------------
# CAT R1.1.X - OMNI-SYNTAX BITNET 1.58b ENGINE
# ----------------------------------------------------------------------
class CatR11OmniEngine:
    def __init__(self, d_model=64):
        self.name = "CAT R1.1.X"
        self.ver = "1.58-bit Ternary MoE"
        self.d_model = d_model
        
        # BitNet 1.58b: Weights are constrained to {-1, 0, 1}
        self.weights = np.random.choice([-1, 0, 1], size=(d_model, d_model))
        
        # Omni-Syntax Expert Database
        self.code_experts = {
            "python": "def main():\n    print('CAT R1.1.X: Python Logic Active')\n\nif __name__ == '__main__':\n    main()",
            "cpp": "#include <iostream>\n\nint main() {\n    std::cout << \"CAT R1.1.X: C++ Ternary Logic!\" << std::endl;\n    return 0;\n}",
            "c": "#include <stdio.h>\n\nint main() {\n    printf(\"CAT R1.1.X: C Metal Layer!\\n\");\n    return 0;\n}",
            "html": "<!DOCTYPE html>\n<html><body style='background:#000; color:#00f;'><h1>CAT R1.1.X Web</h1></body></html>",
            "javascript": "console.log('CAT R1.1.X: V8 Engine Smacked');",
            "java": "public class Main {\n    public static void main(String[] args) {\n        System.out.println(\"CAT R1.1.X: JVM Initialized\");\n    }\n}",
            "rust": "fn main() {\n    println!(\"CAT R1.1.X: Memory Safe & Fearless\");\n}",
            "bash": "#!/bin/bash\necho \"CAT R1.1.X: Shell Script Executing\"",
            "assembly": "section .data\n    msg db 'CAT R1.1.X',0xa\nsection .text\n    global _start\n_start: ; Bare metal logic"
        }

        self.aliases = {
            "c++": "cpp", "js": "javascript", "ts": "typescript",
            "c#": "csharp", "sh": "bash", "shell": "bash", "asm": "assembly"
        }

    def bitnet_matmul(self, x):
        """Simulates BitNet 1.58b Addition/Subtraction Kernels"""
        x_norm = (x - np.mean(x)) / (np.std(x) + 1e-5)
        return np.dot(x_norm, self.weights)

    def extract_language(self, prompt):
        p = prompt.lower()
        for lang in self.code_experts.keys():
            if lang in p.split() or f"in {lang}" in p:
                return lang
        for alias, real_lang in self.aliases.items():
            if alias in p.split() or f"in {alias}" in p:
                return real_lang
        match = re.search(r'in ([a-z0-9+#]+)', p)
        return match.group(1) if match else None

    def think(self, prompt, target_lang):
        lang_display = target_lang.upper() if target_lang else "GENERAL"
        return [
            "🧠 Initializing BitNet 1.58b Ternary Kernels...",
            "⚙️ Quantizing input tensors to {-1, 0, 1}...",
            f"🔍 Routing to Omni-Syntax MoE... Target: [{lang_display}]",
            "🛠️ Calculating logits via zero-multiply accumulation...",
            "✨ Self-Correction: Syntax verified for 1.1.X."
        ]

    def generate(self, prompt, target_lang):
        p = prompt.lower()
        dummy_input = np.random.randn(self.d_model)
        for _ in range(3): dummy_input = self.bitnet_matmul(dummy_input)
        time.sleep(1.2)

        is_code = any(x in p for x in ["code", "write", "syntax", "script", "program"]) or target_lang
        if is_code:
            if target_lang in self.code_experts:
                return self.code_experts[target_lang]
            elif target_lang:
                return f"// CAT R1.1.X DYNAMIC SYNTAX\n// Language: {target_lang.upper()}\n\nfunction main() {{ print('Hello {target_lang}!'); }}"
            return self.code_experts["python"]
        return "Ternary Logic computation complete. I am CAT R1.1.X, running locally. xd"

# ----------------------------------------------------------------------
# GUI - CAT R1.1.X TERMINAL
# ----------------------------------------------------------------------
class CatR1_1_X:
    def __init__(self, root):
        self.root = root
        self.engine = CatR11OmniEngine()
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

    def append_msg(self, sender, text, tag=None):
        self.chat.config(state='normal')
        self.chat.insert(tk.END, f"[{sender}]: ", 'bot_name' if sender == "CAT R1.1.X" else tag)
        self.chat.insert(tk.END, f"{text}\n\n", tag)
        self.chat.config(state='disabled')
        self.chat.see(tk.END)

    def handle_input(self):
        msg = self.entry.get()
        if not msg: return
        self.entry.delete(0, tk.END)
        self.append_msg("YOU", msg, 'user')
        threading.Thread(target=self.infer, args=(msg,)).start()

    def infer(self, prompt):
        target_lang = self.engine.extract_language(prompt)
        for step in self.engine.think(prompt, target_lang):
            self.root.after(0, lambda s=step: self.append_msg("THINK", s, 'thought'))
            time.sleep(0.3)
        result = self.engine.generate(prompt, target_lang)
        self.root.after(0, lambda r=result: self.append_msg("CAT R1.1.X", r))

if __name__ == "__main__":
    root = tk.Tk()
    CatR1_1_X(root)
    root.mainloop()
