import tkinter as tk
from tkinter import ttk, messagebox
from model import predict_stress


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#0f172a",          # slate-900 (app background)
    "surface": "#1e293b",     # slate-800 (cards)
    "surface_alt": "#273549", # slightly lighter card / input bg
    "border": "#334155",      # slate-700
    "text": "#f1f5f9",        # slate-100
    "text_muted": "#94a3b8",  # slate-400
    "accent": "#6366f1",      # indigo-500
    "accent_hover": "#4f46e5",# indigo-600
    "danger": "#ef4444",      # red-500
    "danger_bg": "#3b1219",
    "success": "#22c55e",     # green-500
    "success_bg": "#102a1c",
    "secondary": "#334155",
    "secondary_hover": "#475569",
}

FONT_TITLE = ("Segoe UI", 20, "bold")
FONT_SUBTITLE = ("Segoe UI", 10)
FONT_LABEL = ("Segoe UI", 10, "bold")
FONT_BODY = ("Segoe UI", 11)
FONT_RESULT = ("Segoe UI", 22, "bold")
FONT_CONF = ("Segoe UI", 11)
FONT_BUTTON = ("Segoe UI", 10, "bold")


class HoverButton(tk.Button):
    """A flat tkinter button with a hover color transition for a modern feel."""

    def __init__(self, master, bg, hover_bg, fg="white", **kwargs):
        super().__init__(
            master,
            bg=bg,
            fg=fg,
            activebackground=hover_bg,
            activeforeground=fg,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=FONT_BUTTON,
            padx=18,
            pady=10,
            **kwargs,
        )
        self._bg = bg
        self._hover_bg = hover_bg
        self.bind("<Enter>", lambda e: self.config(bg=self._hover_bg))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg))


class StressApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mental Stress Predictor")
        self.root.geometry("720x620")
        self.root.minsize(800, 600)
        self.root.config(bg=COLORS["bg"])

        # ttk's "default" theme on Windows ignores custom colors for widgets
        # like Progressbar, so force "clam" which honors them everywhere.
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Everything lives inside a scrollable canvas so the Result card is
        # always reachable even if the window is small or gets resized down.
        container = tk.Frame(self.root, bg=COLORS["bg"])
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.scroll_frame = tk.Frame(canvas, bg=COLORS["bg"])

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas_window = canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Keep the inner frame the same width as the canvas viewport.
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfig(canvas_window, width=e.width),
        )

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.root = self.scroll_frame  # build sections inside the scroll frame

        self._build_header()
        self._build_input_card()
        self._build_actions()
        self._build_result_card()
        self._build_footer()

    # ------------------------------------------------------------------
    # UI sections
    # ------------------------------------------------------------------
    def _build_header(self):
        header = tk.Frame(self.root, bg=COLORS["bg"])
        header.pack(fill="x", padx=30, pady=(28, 10))

        tk.Label(
            header, text="Mental Stress Predictor",
            font=FONT_TITLE, fg=COLORS["text"], bg=COLORS["bg"],
        ).pack(anchor="w")

        tk.Label(
            header, text="Paste or type a piece of text to analyze its emotional tone.",
            font=FONT_SUBTITLE, fg=COLORS["text_muted"], bg=COLORS["bg"],
        ).pack(anchor="w", pady=(4, 0))

    def _build_input_card(self):
        card = tk.Frame(self.root, bg=COLORS["surface"])
        card.pack(fill="x", padx=30, pady=10)

        inner = tk.Frame(card, bg=COLORS["surface"])
        inner.pack(fill="x", padx=20, pady=18)

        tk.Label(
            inner, text="YOUR TEXT", font=FONT_LABEL,
            fg=COLORS["text_muted"], bg=COLORS["surface"],
        ).pack(anchor="w", pady=(0, 8))

        text_wrap = tk.Frame(inner, bg=COLORS["border"])
        text_wrap.pack(fill="x")

        self.text_box = tk.Text(
            text_wrap, height=7, font=FONT_BODY, wrap="word",
            bg=COLORS["surface_alt"], fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="flat", bd=0, padx=12, pady=10,
            highlightthickness=0,
        )
        self.text_box.pack(fill="x", padx=1, pady=1)
        self.text_box.insert("1.0", "")
        self._placeholder = "e.g. I have so much to do and no time to finish it all..."
        self._add_placeholder()

    def _add_placeholder(self):
        self.text_box.insert("1.0", self._placeholder)
        self.text_box.config(fg=COLORS["text_muted"])
        self.text_box.bind("<FocusIn>", self._clear_placeholder)
        self.text_box.bind("<FocusOut>", self._restore_placeholder)

    def _clear_placeholder(self, event=None):
        if self.text_box.get("1.0", "end-1c") == self._placeholder:
            self.text_box.delete("1.0", tk.END)
            self.text_box.config(fg=COLORS["text"])

    def _restore_placeholder(self, event=None):
        if not self.text_box.get("1.0", "end-1c").strip():
            self._add_placeholder()

    def _build_actions(self):
        actions = tk.Frame(self.root, bg=COLORS["bg"])
        actions.pack(fill="x", padx=30, pady=(4, 6))

        HoverButton(
            actions, text="Analyze Text",
            bg=COLORS["accent"], hover_bg=COLORS["accent_hover"],
            command=self.predict,
        ).pack(side="left")

        HoverButton(
            actions, text="Clear",
            bg=COLORS["secondary"], hover_bg=COLORS["secondary_hover"],
            command=self.clear,
        ).pack(side="left", padx=(10, 0))

    def _build_result_card(self):
        self.result_card = tk.Frame(self.root, bg=COLORS["surface"])
        self.result_card.pack(fill="x", padx=30, pady=10)

        inner = tk.Frame(self.result_card, bg=COLORS["surface"])
        inner.pack(fill="x", padx=20, pady=20)

        tk.Label(
            inner, text="RESULT", font=FONT_LABEL,
            fg=COLORS["text_muted"], bg=COLORS["surface"],
        ).pack(anchor="w", pady=(0, 10))

        # Badge that holds the predicted label
        self.badge = tk.Label(
            inner, text="No prediction yet", font=FONT_RESULT,
            fg=COLORS["text_muted"], bg=COLORS["surface"],
            padx=16, pady=8,
        )
        self.badge.pack(anchor="w")

        # Confidence row
        conf_row = tk.Frame(inner, bg=COLORS["surface"])
        conf_row.pack(fill="x", pady=(16, 0))

        self.conf_label = tk.Label(
            conf_row, text="", font=FONT_CONF,
            fg=COLORS["text_muted"], bg=COLORS["surface"],
        )
        self.conf_label.pack(anchor="w", pady=(0, 6))

        style = ttk.Style()
        style.configure(
            "Conf.Horizontal.TProgressbar",
            troughcolor=COLORS["surface_alt"],
            bordercolor=COLORS["surface"],
            background=COLORS["accent"],
            lightcolor=COLORS["accent"],
            darkcolor=COLORS["accent"],
            thickness=10,
        )
        self.conf_bar = ttk.Progressbar(
            conf_row, style="Conf.Horizontal.TProgressbar",
            orient="horizontal", mode="determinate", maximum=100,
        )
        self.conf_bar.pack(fill="x")

    def _build_footer(self):
        tk.Label(
            self.root, text="Your text is analyzed locally and is not stored.",
            font=("Segoe UI", 8), fg=COLORS["text_muted"], bg=COLORS["bg"],
        ).pack(side="bottom", pady=14)

    # ------------------------------------------------------------------
    # Logic
    # ------------------------------------------------------------------
    def predict(self):
        text = self.text_box.get("1.0", tk.END).strip()

        if not text or text == self._placeholder:
            messagebox.showwarning("Warning", "Please enter some text first.")
            return

        label, confidence = predict_stress(text)
        pct = int(confidence * 100)
        is_stressed = label == "Stressed"

        badge_fg = COLORS["danger"] if is_stressed else COLORS["success"]
        badge_bg = COLORS["danger_bg"] if is_stressed else COLORS["success_bg"]

        self.badge.config(text=label, fg=badge_fg, bg=badge_bg)
        self.conf_label.config(text=f"Confidence  ·  {pct}%")
        self.conf_bar.config(value=pct)

    def clear(self):
        self.text_box.delete("1.0", tk.END)
        self._add_placeholder()
        self.badge.config(text="No prediction yet", fg=COLORS["text_muted"], bg=COLORS["surface"])
        self.conf_label.config(text="")
        self.conf_bar.config(value=0)


if __name__ == "__main__":
    root = tk.Tk()
    app = StressApp(root)
    root.mainloop()