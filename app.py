# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 15:18:52 2026

@author: 90507
"""

import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import networkx as nx
import numpy as np
from pathlib import Path
from clang.cindex import Config, Index, CursorKind
from scipy.optimize import linear_sum_assignment
from networkx.drawing.nx_pydot import write_dot
import graphviz
from collections import Counter
from PIL import Image, ImageTk
from io import BytesIO
import ast


# --- Clang Yapılandırması ---
bin_dir = os.path.join(os.environ["CONDA_PREFIX"], "Library", "bin")
os.add_dll_directory(bin_dir)
libclang = os.path.join(bin_dir, "libclang-13.dll")

if not Config.loaded:
    Config.set_library_file(libclang)

INTERESTING_KINDS = {
    CursorKind.FUNCTION_DECL, 
    CursorKind.IF_STMT, 
    CursorKind.FOR_STMT, 
    CursorKind.WHILE_STMT, 
    CursorKind.DO_STMT,
    CursorKind.SWITCH_STMT, 
    CursorKind.CASE_STMT, 
    CursorKind.DEFAULT_STMT,
    CursorKind.CALL_EXPR,
    CursorKind.RETURN_STMT,
    CursorKind.BINARY_OPERATOR
}
INTERESTING_PY_NODES = (
    ast.FunctionDef,
    ast.If,
    ast.For,
    ast.While,
    ast.Call,
    ast.Return,
    ast.BinOp,
    ast.Compare
)


def visualize_graph(graph, output_name):
    dot_path = f"{output_name}.dot"
    png_path = f"{output_name}.png"


    write_dot(graph, dot_path)


    dot = graphviz.Source.from_file(dot_path)
    dot.render(output_name, format="png", cleanup=True)

    print(f"Graph saved as {png_path}")

def make_label(node):
    k = node.kind.name


    if node.kind == CursorKind.FUNCTION_DECL:

        return "FUNCTION"

    if node.kind == CursorKind.CALL_EXPR:

        return "CALL"

    if node.kind == CursorKind.BINARY_OPERATOR:
        toks = [t.spelling for t in node.get_tokens()]
        for op in ["+", "-", "*", "/", "%", "==", "!=", "<", "<=", ">", ">=", "&&", "||", "=", "+=", "-=", "*=", "/="]:
            if op in toks:
                return f"BINOP:{op}"
        return "BINOP"

    return k



def _child_label_multiset(g, node_id):

    labs = []
    for v in g.successors(node_id):
        labs.append(g.nodes[v].get("label", ""))
    return Counter(labs)

def _degree(g, node_id):
    return g.in_degree(node_id), g.out_degree(node_id)

def _multiset_jaccard(c1: Counter, c2: Counter) -> float:
    if not c1 and not c2:
        return 1.0
    inter = sum((c1 & c2).values())
    union = sum((c1 | c2).values())
    return inter / union if union else 1.0

def build_graph(node, graph=None, root_path=None, parent_interesting_id=None):
    if graph is None:
        graph = nx.DiGraph()

    curr_file = str(node.location.file) if node.location.file else None
    if curr_file and root_path and Path(curr_file).name != Path(root_path).name:
        return graph

    current_parent = parent_interesting_id


    if node.kind in INTERESTING_KINDS:
        node_id = str(node.hash)
        graph.add_node(node_id, label=make_label(node))

        if parent_interesting_id is not None:
            graph.add_edge(parent_interesting_id, node_id)

        current_parent = node_id  


    for child in node.get_children():
        build_graph(child, graph, root_path, current_parent)

    return graph
def build_graph_py(node, graph=None, parent_interesting_id=None):
    if graph is None:
        graph = nx.DiGraph()

    current_parent = parent_interesting_id

    if isinstance(node, INTERESTING_PY_NODES):
        node_id = str(id(node))

        label = type(node).__name__
        if isinstance(node, ast.BinOp):
            label = "BINOP"
        elif isinstance(node, ast.Call):
            label = "CALL"
        elif isinstance(node, ast.FunctionDef):
            label = "FUNCTION"

        graph.add_node(node_id, label=label)

        if parent_interesting_id is not None:
            graph.add_edge(parent_interesting_id, node_id)

        current_parent = node_id

    for child in ast.iter_child_nodes(node):
        build_graph_py(child, graph, current_parent)

    return graph


def calculate_riesen_burke_similarity_symmetric(
    g1, g2,
    c_ins=1.0, c_del=1.0,
    w_label=1.0, w_deg=0.15, w_child=0.9,
    label_sub_cost=1.0,
    match_cost_threshold=0.55,
    require_same_label=True
):
    
    nodes1 = list(g1.nodes())
    nodes2 = list(g2.nodes())
    n, m = len(nodes1), len(nodes2)

    if n == 0 and m == 0:
        return 1.0, []
    if n == 0 or m == 0:
        return 0.0, []

    lab1 = [g1.nodes[u].get("label", "") for u in nodes1]
    lab2 = [g2.nodes[v].get("label", "") for v in nodes2]

    deg1 = [_degree(g1, u) for u in nodes1]
    deg2 = [_degree(g2, v) for v in nodes2]

    child1 = [_child_label_multiset(g1, u) for u in nodes1]
    child2 = [_child_label_multiset(g2, v) for v in nodes2]

    size = n + m
    BIG = 1e9
    cost = np.full((size, size), BIG, dtype=float)

    sub_cap = c_del + c_ins  

    for i in range(n):
        in1, out1 = deg1[i]
        for j in range(m):
            in2, out2 = deg2[j]

            c_lab = 0.0 if lab1[i] == lab2[j] else label_sub_cost
            c_d = (abs(in1 - in2) + abs(out1 - out2)) / 4.0
            jac = _multiset_jaccard(child1[i], child2[j])
            c_ch = 1.0 - jac

            sub = (w_label * c_lab) + (w_deg * c_d) + (w_child * c_ch)
            cost[i, j] = min(sub, sub_cap)


    for i in range(n):
        cost[i, m + i] = c_del

    for j in range(m):
        cost[n + j, j] = c_ins

    cost[n:, m:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost)
    dist = float(cost[row_ind, col_ind].sum())


    max_cost = max(n, m) * (c_del + c_ins)
    sim = 1.0 - (dist / max_cost) if max_cost > 0 else 0.0
    sim = max(0.0, min(1.0, sim))


    matches = []
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m:
            pair_cost = float(cost[r, c])
            if pair_cost <= match_cost_threshold:
                if (not require_same_label) or (lab1[r] == lab2[c]):
                    matches.append((nodes1[r], nodes2[c], pair_cost))

    return sim, matches


def color_matches_red(g1, g2, matches, color="red"):
    matched1 = {u for (u, v, _) in matches}
    matched2 = {v for (u, v, _) in matches}

    # Node styling (Graphviz attributes)
    for u in matched1:
        g1.nodes[u]["color"] = color
        g1.nodes[u]["fontcolor"] = color
        g1.nodes[u]["penwidth"] = "2"

    for v in matched2:
        g2.nodes[v]["color"] = color
        g2.nodes[v]["fontcolor"] = color
        g2.nodes[v]["penwidth"] = "2"

    # OPTIONAL: also color edges between matched nodes
    def mark_edges(g, matched_set):
        for a, b in g.edges():
            if a in matched_set and b in matched_set:
                g.edges[a, b]["color"] = color
                g.edges[a, b]["penwidth"] = "2"

    mark_edges(g1, matched1)
    mark_edges(g2, matched2)

def compare_codes(f1_path, f2_path):
    index = Index.create()

    g1 = build_graph(index.parse(f1_path).cursor, root_path=f1_path)
    g2 = build_graph(index.parse(f2_path).cursor, root_path=f2_path)

    print(f"Graf 1 Düğüm Sayısı: {g1.number_of_nodes()}")
    print(f"Graf 2 Düğüm Sayısı: {g2.number_of_nodes()}")

    sim, matches = calculate_riesen_burke_similarity_symmetric(g1, g2)

    color_matches_red(g1, g2, matches, color="red")
    if g1.number_of_nodes() == 0 or g2.number_of_nodes() == 0:
        raise RuntimeError("Graflar oluşturulamadı. Dosya yollarını kontrol edin.")
    visualize_graph(g1, "graph_code_1")
    visualize_graph(g2, "graph_code_2")
    sim, matches = calculate_riesen_burke_similarity_symmetric(g1, g2)
    print("\n--- Sonuç ---")
    print(f"Benzerlik Oranı: %{sim * 100:.2f}")
    
    return sim
def compare_codes_py(f1_path, f2_path):
    with open(f1_path, "r", encoding="utf-8") as f:
        tree1 = ast.parse(f.read())

    with open(f2_path, "r", encoding="utf-8") as f:
        tree2 = ast.parse(f.read())

    g1 = build_graph_py(tree1)
    g2 = build_graph_py(tree2)

    print(f"Python Graph 1 nodes: {g1.number_of_nodes()}")
    print(f"Python Graph 2 nodes: {g2.number_of_nodes()}")

    sim, matches =calculate_riesen_burke_similarity_symmetric(g1, g2)

    color_matches_red(g1, g2, matches)

    visualize_graph(g1, "graph_code_1")
    visualize_graph(g2, "graph_code_2")

    print(f"Python Similarity: %{sim * 100:.2f}")
    return sim

class ZoomableCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="white", **kwargs)

        self.img_orig = None     
        self.img_tk = None       
        self.image_id = None

        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 6.0
        self.bind("<ButtonPress-1>", self._start_pan)
        self.bind("<B1-Motion>", self._do_pan)

        self.bind("<MouseWheel>", self._zoom)
        self.bind("<Button-4>", self._zoom)
        self.bind("<Button-5>", self._zoom)
    def load_image(self, path):
        with open(path, "rb") as f:
             data = f.read()
        self.img_orig = Image.open(BytesIO(data)).convert("RGBA")
        self.scale = 1.0
        self.delete("all") 
        self.image_id = None          
        self.img_tk = None           

        self._redraw(center=True)



    def _redraw(self, center=False):
        if self.img_orig is None:
            return

        w, h = self.img_orig.size
        new_size = (int(w * self.scale), int(h * self.scale))

        img = self.img_orig.resize(new_size, Image.LANCZOS)
        self.img_tk = ImageTk.PhotoImage(img)

        if self.image_id is None:
            self.image_id = self.create_image(0, 0, anchor="nw", image=self.img_tk)
        else:
            self.itemconfig(self.image_id, image=self.img_tk)

        self.config(scrollregion=self.bbox(self.image_id))

        if center:
            self._center_image()

    def _center_image(self):
        self.update_idletasks()
        cw = self.winfo_width()
        ch = self.winfo_height()
        iw = self.img_tk.width()
        ih = self.img_tk.height()

        x = max((cw - iw) // 2, 0)
        y = max((ch - ih) // 2, 0)
        self.coords(self.image_id, x, y)

    def _start_pan(self, event):
        self.scan_mark(event.x, event.y)

    def _do_pan(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def _zoom(self, event):
        if self.img_orig is None:
            return

        if event.delta > 0 or getattr(event, "num", None) == 4:
            factor = 1.1
        else:
            factor = 0.9

        new_scale = self.scale * factor
        if not (self.min_scale <= new_scale <= self.max_scale):
            return

        self.scale = new_scale
        self._redraw()

class CompareGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("C Code Similarity (AST Graph)")
        self.geometry("1200x700")

        self.file1_var = tk.StringVar()
        self.file2_var = tk.StringVar()
        self.file3_var = tk.StringVar()
        self.file4_var = tk.StringVar()
        self.result_var = tk.StringVar(value="Benzerlik: -")
        self.status_var = tk.StringVar(value="Hazır")

        # Keep image refs here (important!)
        self.img1_tk = None
        self.img2_tk = None

        pad = {"padx": 10, "pady": 6}

        top = tk.Frame(self)
        top.pack(fill="x")

        tk.Label(top, text="Dosya 1 (.c):").grid(row=0, column=0, sticky="w", **pad)
        tk.Entry(top, textvariable=self.file1_var, width=60).grid(row=0, column=1, **pad)
        tk.Button(top, text="Gözat...", command=self.browse1).grid(row=0, column=2, **pad)
        
        tk.Label(top, text="Dosya 1 (.py):").grid(row=0, column=3, sticky="w", **pad)
        tk.Entry(top, textvariable=self.file3_var, width=60).grid(row=0, column=4, **pad)
        tk.Button(top, text="Gözat...", command=self.browse3).grid(row=0, column=5, **pad)
        
        tk.Label(top, text="Dosya 2 (.c):").grid(row=1, column=0, sticky="w", **pad)
        tk.Entry(top, textvariable=self.file2_var, width=60).grid(row=1, column=1, **pad)
        tk.Button(top, text="Gözat...", command=self.browse2).grid(row=1, column=2, **pad)
        
        tk.Label(top, text="Dosya 2 (.py):").grid(row=1, column=3, sticky="w", **pad)
        tk.Entry(top, textvariable=self.file4_var, width=60).grid(row=1, column=4, **pad)
        tk.Button(top, text="Gözat...", command=self.browse4).grid(row=1, column=5, **pad)

        tk.Button(top, text="Karşılaştır", command=self.run_compare).grid(row=2, column=1, sticky="e", **pad)
        tk.Button(top, text="Karşılaştır", command=self.run_compare1).grid(row=2, column=4, sticky="e", **pad)
        
        tk.Label(top, textvariable=self.result_var, font=("Segoe UI", 12, "bold")).grid(
            row=3, column=0, columnspan=3, sticky="w", **pad
        )
        tk.Label(top, textvariable=self.status_var).grid(row=4, column=0, columnspan=3, sticky="w", **pad)
        img_area = tk.Frame(self)
        img_area.pack(fill="both", expand=True)

        self.left_panel = tk.LabelFrame(img_area, text="Graph 1", padx=8, pady=8)
        self.left_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.right_panel = tk.LabelFrame(img_area, text="Graph 2", padx=8, pady=8)
        self.right_panel.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        self.canvas1 = ZoomableCanvas(self.left_panel)
        self.canvas1.pack(fill="both", expand=True)

        self.canvas2 = ZoomableCanvas(self.right_panel)
        self.canvas2.pack(fill="both", expand=True)

        self.bind("<Configure>", self._on_resize)

    def browse1(self):
        path = filedialog.askopenfilename(filetypes=[("C files", "*.c"), ("All files", "*.*")])
        if path:
            self.file1_var.set(path)

    def browse2(self):
        path = filedialog.askopenfilename(filetypes=[("C files", "*.c"), ("All files", "*.*")])
        if path:
            self.file2_var.set(path)
    def browse3(self):
        path = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if path:
            self.file3_var.set(path)
    def browse4(self):
        path = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if path:
            self.file4_var.set(path)
    def set_busy(self, busy: bool):
        self.status_var.set("Çalışıyor..." if busy else "Hazır")
        self.update_idletasks()

    def run_compare(self):
        f1 = self.file1_var.get().strip()
        f2 = self.file2_var.get().strip()

        if not f1 or not f2:
            messagebox.showwarning("Eksik bilgi", "Lütfen iki adet .c dosyası seçin.")
            return

        def worker():
            try:
                self.after(0, lambda: self.set_busy(True))
                sim = compare_codes(f1, f2)  # make sure compare_codes returns sim (float 0..1)
                self.after(0, lambda: self.result_var.set(f"Benzerlik: %{sim * 100:.2f}"))
                self.after(0, lambda: self.status_var.set("Bitti. PNG önizlemeleri güncelleniyor..."))
                self.after(0, self.refresh_previews)
                self.after(0, lambda: self.status_var.set("Bitti."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Hata", str()))
                self.after(0, lambda: self.status_var.set("Hata oluştu."))
            finally:
                self.after(0, lambda: self.set_busy(False))

        threading.Thread(target=worker, daemon=True).start()
    def run_compare1(self):
        f3 = self.file3_var.get().strip()
        f4 = self.file4_var.get().strip()

        if not f3 or not f4:
            messagebox.showwarning("Eksik bilgi", "Lütfen iki adet .py dosyası seçin.")
            return

        def worker():
            try:
                self.after(0, lambda: self.set_busy(True))
                sim = compare_codes_py(f3, f4)  
                self.after(0, lambda: self.result_var.set(f"Benzerlik: %{sim * 100:.2f}"))
                self.after(0, lambda: self.status_var.set("Bitti. PNG önizlemeleri güncelleniyor..."))
                self.after(0, self.refresh_previews)
                self.after(0, lambda: self.status_var.set("Bitti."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Hata", str()))
                self.after(0, lambda: self.status_var.set("Hata oluştu."))
            finally:
                self.after(0, lambda: self.set_busy(False))

        threading.Thread(target=worker, daemon=True).start()
    def refresh_previews(self):
     if os.path.exists("graph_code_1.png"):
        self.canvas1.load_image("graph_code_1.png")

     if os.path.exists("graph_code_2.png"):
        self.canvas2.load_image("graph_code_2.png")


    def _panel_target_size(self, panel: tk.Widget):
        # a safe usable area inside the panel
        w = max(panel.winfo_width() - 30, 200)
        h = max(panel.winfo_height() - 60, 200)
        return w, h

    def _load_into_label(self, path: str, which: int):
        import os
        if not os.path.exists(path):
            if which == 1:
                self.img1_label.config(text=f"{path} not found")
            else:
                self.img2_label.config(text=f"{path} not found")
            return

        panel = self.left_panel if which == 1 else self.right_panel
        target_w, target_h = self._panel_target_size(panel)

        img = Image.open(path)
        img.thumbnail((target_w, target_h), Image.LANCZOS)  

        img_tk = ImageTk.PhotoImage(img)

        if which == 1:
            self.img1_tk = img_tk
            self.img1_label.config(image=self.img1_tk, text="")
        else:
            self.img2_tk = img_tk
            self.img2_label.config(image=self.img2_tk, text="")

    def _on_resize(self, event):

        if event.widget is self:

            if self.img1_tk or self.img2_tk:
                self.refresh_previews()


if __name__ == "__main__":
    app = CompareGUI()
    app.mainloop()
