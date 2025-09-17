# SUBMITTED BY THE STUDENTS:         OMAR EHAB ABUDAYYEH             - 2136037
#                                   AHMAD Mohamad AL-HAJ             - 2141147
#                                   MUHAMMAD MUSTAFA AL-MASHAYEKH   - 2138237

import tkinter as tk
import importlib.util
import os
import io
import sys


class AIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Interface")
        self.path = tk.StringVar()
        self.module = None
        self.is_gpu_module = False  # Track if loaded module is GPU version

        # CPU/GPU Buttons
        self.cpu_btn = tk.Button(root, text="CPU", command=self.load_cpu)
        self.cpu_btn.place(x=20, y=20, width=80, height=40)

        self.gpu_btn = tk.Button(root, text="GPU", command=self.load_gpu)
        self.gpu_btn.place(x=20, y=80, width=80, height=40)

        # Output area with scrollbar
        self.output_text = tk.Text(root, bg="lightblue", wrap="word")
        self.output_text.place(x=120, y=20, width=560, height=260)

        scrollbar = tk.Scrollbar(root, command=self.output_text.yview)
        scrollbar.place(x=680, y=20, height=260)
        self.output_text.config(yscrollcommand=scrollbar.set)

        # Path input field (fully visible)
        tk.Label(root, text="Path:").place(x=120, y=290)
        self.path_entry = tk.Entry(root, textvariable=self.path)
        self.path_entry.place(x=170, y=290, width=350, height=30)

        # Train/Test buttons
        self.train_btn = tk.Button(root, text="Train", command=self.train_model)
        self.train_btn.place(x=540, y=290, width=60, height=30)

        self.test_btn = tk.Button(root, text="Test", command=self.test_model)
        self.test_btn.place(x=610, y=290, width=60, height=30)

    def load_module(self, file_path):
        if not os.path.exists(file_path):
            self.display_output(f"File not found: {file_path}")
            return None
        spec = importlib.util.spec_from_file_location("ai_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def load_cpu(self):
        path = os.path.join(os.path.dirname(__file__), "Sign_detection_Team3.py")
        self.module = self.load_module(path)
        self.is_gpu_module = False
        if self.module:
            self.display_output("CPU Script Loaded")

    def load_gpu(self):
        path = os.path.join(os.path.dirname(__file__), "Sign_detection_gpu.py")
        self.module = self.load_module(path)
        self.is_gpu_module = True
        if self.module:
            self.display_output("GPU Script Loaded")

    def train_model(self):
        self.clear_output()
        if not self.module:
            self.display_output("No module loaded.")
            return

        if self.is_gpu_module:
            # For GPU, call train_and_save_model(path)
            if hasattr(self.module, 'train_and_save_model'):
                self.capture_output(lambda: self.module.train_and_save_model(self.path.get()))
            else:
                self.display_output("train_and_save_model function not found in GPU module.")
        else:
            # For CPU, call train(path)
            if hasattr(self.module, 'train'):
                self.capture_output(lambda: self.module.train(self.path.get()))
            else:
                self.display_output("train function not found in CPU module.")

    def test_model(self):
        self.clear_output()
        if not self.module:
            self.display_output("No module loaded.")
            return

        if self.is_gpu_module:
            # For GPU, call run_prediction(path)
            if hasattr(self.module, 'run_prediction'):
                self.capture_output(lambda: self.module.run_prediction(self.path.get()))
            else:
                self.display_output("run_prediction function not found in GPU module.")
        else:
            # For CPU, try existing test functions
            if hasattr(self.module, 'test'):
                self.capture_output(lambda: self.module.test(self.path.get()))
            elif hasattr(self.module, 'test_button'):
                self.capture_output(lambda: self.module.test_button("parameters.npz", self.path.get()))
            else:
                self.display_output("Test function not found in CPU module.")

    def display_output(self, message):
        self.output_text.insert(tk.END, message + "\n")
        self.output_text.see(tk.END)

    def clear_output(self):
        self.output_text.delete(1.0, tk.END)

    def capture_output(self, func):
        old_stdout = sys.stdout
        redirected_output = sys.stdout = io.StringIO()
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            sys.stdout = old_stdout
            output = redirected_output.getvalue()
            self.display_output(output)


# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("720x340")
    app = AIApp(root)
    root.mainloop()
