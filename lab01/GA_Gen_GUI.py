import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


MAX_POP = 20
MAX_GEN = 20


@dataclass
class Individual:
    chrom: List[int]
    sila: int = 0
    cargo: int = 0
    rodzic: int = 0


def fitness_onemax(chrom: List[int]) -> int:
    return sum(chrom)


def ocena_populacji(pop: List[Individual]) -> None:
    for ind in pop:
        ind.sila = fitness_onemax(ind.chrom)


def sortuj_populacje_jak_qsort(pop: List[Individual]) -> None:
    # malejąco po sila, a przy remisie rosnąco po cargo (deterministycznie)
    pop.sort(key=lambda ind: (-ind.sila, ind.cargo))


def selekcja_ruletka_posortowana(pop: List[Individual]) -> Individual:
    suma = sum(ind.sila for ind in pop)
    if suma == 0:
        chosen = random.choice(pop)
        chosen.rodzic += 1
        return chosen

    traf = random.uniform(0, suma)
    acc = 0.0
    for ind in pop:
        acc += ind.sila
        if traf <= acc:
            ind.rodzic += 1
            return ind
    pop[-1].rodzic += 1
    return pop[-1]


def losuj_punkty_krzyzowania(gen_len: int, punkty: int) -> List[int]:
    pts = {0, gen_len}
    while len(pts) < punkty + 2:
        pts.add(random.randrange(0, gen_len))
    return sorted(pts)


def krzyzowanie_wielopunktowe(p1: Individual, p2: Individual, punkty: int,
                             cargo_counter: int) -> Tuple[Individual, Individual, int]:
    t_pk = losuj_punkty_krzyzowania(MAX_GEN, punkty)

    child0: List[int] = []
    child1: List[int] = []

    for i in range(1, len(t_pk)):
        a, b = t_pk[i - 1], t_pk[i]
        if i % 2 == 0:
            child1.extend(p1.chrom[a:b])
            child0.extend(p2.chrom[a:b])
        else:
            child0.extend(p1.chrom[a:b])
            child1.extend(p2.chrom[a:b])

    cargo_counter += 1
    c0 = Individual(chrom=child0, cargo=cargo_counter)
    cargo_counter += 1
    c1 = Individual(chrom=child1, cargo=cargo_counter)
    c0.sila = fitness_onemax(c0.chrom)
    c1.sila = fitness_onemax(c1.chrom)
    return c0, c1, cargo_counter


def mutacja(pop: List[Individual], pm: float) -> int:
    licznosc = len(pop) * MAX_GEN
    ile = int(pm * licznosc)
    for _ in range(ile):
        traf = random.randrange(0, licznosc)
        w = traf // MAX_GEN
        k = traf % MAX_GEN
        pop[w].chrom[k] ^= 1
    ocena_populacji(pop)
    return ile


def gen_populacji_start(pop_size: int, gen_len: int, cargo_counter: int) -> Tuple[List[Individual], int]:
    pop: List[Individual] = []
    for _ in range(pop_size):
        chrom = [1 if random.random() < 0.5 else 0 for _ in range(gen_len)]
        cargo_counter += 1
        ind = Individual(chrom=chrom, cargo=cargo_counter)
        ind.sila = fitness_onemax(ind.chrom)
        pop.append(ind)
    return pop, cargo_counter


class GASession:
    def __init__(self):
        self.pop: List[Individual] = []
        self.cargo_counter = 0
        self.gen_idx = 0
        self.best_hist: List[float] = []
        self.avg_hist: List[float] = []
        self.last_mut = 0

        self.running = False
        self.pm = 0.003
        self.punkty = 3
        self.max_generations = 500
        self.seed: Optional[int] = 7
        self.reset_rodzic_each_gen = True
        self.elitism = True

    def reset(self):
        if self.seed is not None:
            random.seed(self.seed)
        self.cargo_counter = 0
        self.gen_idx = 0
        self.best_hist.clear()
        self.avg_hist.clear()
        self.last_mut = 0

        self.pop, self.cargo_counter = gen_populacji_start(MAX_POP, MAX_GEN, self.cargo_counter)
        ocena_populacji(self.pop)
        sortuj_populacje_jak_qsort(self.pop)
        self._append_stats()

    def _append_stats(self):
        fits = [ind.sila for ind in self.pop]
        self.best_hist.append(max(fits))
        self.avg_hist.append(sum(fits) / len(fits))

    def done(self) -> bool:
        if any(ind.sila == MAX_GEN for ind in self.pop):
            return True
        if self.gen_idx >= self.max_generations:
            return True
        return False

    def step(self):
        if self.done():
            return

        if self.reset_rodzic_each_gen:
            for ind in self.pop:
                ind.rodzic = 0

        sortuj_populacje_jak_qsort(self.pop)
        best_prev = None
        if self.elitism:
            best_prev = max(self.pop, key=lambda ind: ind.sila)

        C: List[Individual] = []
        for _ in range(0, MAX_POP, 2):
            r0 = selekcja_ruletka_posortowana(self.pop)
            r1 = selekcja_ruletka_posortowana(self.pop)
            tries = 0
            while r1.cargo == r0.cargo and tries < 50:
                r1 = selekcja_ruletka_posortowana(self.pop)
                tries += 1

            c0, c1, self.cargo_counter = krzyzowanie_wielopunktowe(r0, r1, self.punkty, self.cargo_counter)
            C.append(c0)
            C.append(c1)

        self.last_mut = mutacja(C, self.pm)

        if self.elitism and best_prev is not None and C:
            worst_idx = min(range(len(C)), key=lambda i: C[i].sila)
            C[worst_idx] = Individual(chrom=best_prev.chrom.copy(), sila=best_prev.sila, cargo=best_prev.cargo)

        self.pop = C
        ocena_populacji(self.pop)
        sortuj_populacje_jak_qsort(self.pop)

        self.gen_idx += 1
        self._append_stats()


class GAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Algorytm genetyczny (OneMax) — GUI")
        self.geometry("1200x750")

        self.session = GASession()
        self._build_ui()
        self._init_session()

    def _build_ui(self):
        # ----- Panel parametrów
        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Parametry", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        self.var_pm = tk.StringVar(value="0.003")
        self.var_punkty = tk.StringVar(value="3")
        self.var_seed = tk.StringVar(value="7")
        self.var_maxgen = tk.StringVar(value="500")
        self.var_reset_rodzic = tk.BooleanVar(value=True)
        self.var_elitism = tk.BooleanVar(value=True)

        def row(label, var):
            f = ttk.Frame(left)
            f.pack(fill=tk.X, pady=4)
            ttk.Label(f, text=label, width=16).pack(side=tk.LEFT)
            ttk.Entry(f, textvariable=var, width=12).pack(side=tk.LEFT)

        row("pm:", self.var_pm)
        row("punkty:", self.var_punkty)
        row("seed:", self.var_seed)
        row("max_gen:", self.var_maxgen)

        ttk.Checkbutton(left, text="Zeruj 'rodzic' co gen", variable=self.var_reset_rodzic).pack(anchor="w", pady=6)
        ttk.Checkbutton(left, text="Elityzm", variable=self.var_elitism).pack(anchor="w", pady=6)

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=8)

        ttk.Button(btns, text="Reset", command=self.on_reset).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="Krok", command=self.on_step).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="Start", command=self.on_start).pack(fill=tk.X, pady=3)
        ttk.Button(btns, text="Stop", command=self.on_stop).pack(fill=tk.X, pady=3)

        self.lbl_status = ttk.Label(left, text="Status: -", wraplength=250)
        self.lbl_status.pack(anchor="w", pady=10)

        # ----- Prawa strona: tabela + wykresy
        right = ttk.Frame(self, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # tabela
        ttk.Label(right, text="Tabela populacji (Memo)", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        columns = ("lp", "chrom", "sila", "cargo", "rodzic")
        self.tree = ttk.Treeview(right, columns=columns, show="headings", height=12)
        for c, w in zip(columns, (40, 420, 60, 70, 70)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="center")
        self.tree.pack(fill=tk.X)

        # wykresy
        fig = Figure(figsize=(7.5, 4.5), dpi=100)
        fig.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.08,
        hspace=0.75   # <-- kluczowe (było za małe)
        )
        self.ax_prog = fig.add_subplot(211)
        self.ax_heat = fig.add_subplot(212)

        self.canvas = FigureCanvasTkAgg(fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)

        self.line_best, = self.ax_prog.plot([], [], linewidth=2, label="best")
        self.line_avg, = self.ax_prog.plot([], [], linewidth=2, label="avg")
        self.ax_prog.set_title("Postęp optymalizacji")
        self.ax_prog.set_xlabel("Generacja")
        self.ax_prog.set_ylabel("Fitness")
        self.ax_prog.set_ylim(0, MAX_GEN)
        self.ax_prog.grid(True)
        self.ax_prog.legend()

        self.im = self.ax_heat.imshow(np.zeros((MAX_POP, MAX_GEN)), aspect="auto", vmin=0, vmax=1, interpolation="nearest")
        self.ax_heat.set_title(
        "Populacja (0/1): wiersze=osobniki, kolumny=geny",
        pad=15,        # <-- dodatkowy odstęp od osi
        fontsize=10
        )
        self.ax_heat.set_xlabel("Gen")
        self.ax_heat.set_ylabel("Osobnik")

    def _init_session(self):
        self.session.reset()
        self.refresh_all()

    def read_params_into_session(self) -> bool:
        try:
            self.session.pm = float(self.var_pm.get())
            self.session.punkty = int(self.var_punkty.get())
            self.session.max_generations = int(self.var_maxgen.get())

            s = self.var_seed.get().strip()
            self.session.seed = int(s) if s else None

            self.session.reset_rodzic_each_gen = bool(self.var_reset_rodzic.get())
            self.session.elitism = bool(self.var_elitism.get())

            if not (0.0 <= self.session.pm <= 1.0):
                raise ValueError("pm musi być w [0,1]")
            if not (1 <= self.session.punkty <= 10):
                raise ValueError("punkty krzyżowania powinny być >=1 (sensownie 1..10)")
            if self.session.max_generations < 1:
                raise ValueError("max_gen musi być >= 1")

            return True
        except Exception as e:
            messagebox.showerror("Błąd parametrów", str(e))
            return False

    def refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, ind in enumerate(self.session.pop):
            chrom = "".join(map(str, ind.chrom))
            self.tree.insert("", "end", values=(i, chrom, ind.sila, ind.cargo, ind.rodzic))

    def refresh_plots(self):
        gens = np.arange(len(self.session.best_hist))
        self.line_best.set_data(gens, self.session.best_hist)
        self.line_avg.set_data(gens, self.session.avg_hist)
        self.ax_prog.set_xlim(0, max(1, len(self.session.best_hist) - 1))

        mat = np.array([ind.chrom for ind in self.session.pop], dtype=int)
        self.im.set_data(mat)

        self.canvas.draw_idle()

    def refresh_status(self):
        best = max(ind.sila for ind in self.session.pop)
        avg = sum(ind.sila for ind in self.session.pop) / len(self.session.pop)
        self.lbl_status.config(
            text=f"Generacja: {self.session.gen_idx}\n"
                 f"best={best}  avg={avg:.2f}\n"
                 f"mutacje (ostatni krok)={self.session.last_mut}\n"
                 f"STOP={self.session.done()}"
        )

    def refresh_all(self):
        self.refresh_table()
        self.refresh_plots()
        self.refresh_status()

    def on_reset(self):
        if not self.read_params_into_session():
            return
        self.session.reset()
        self.refresh_all()

    def on_step(self):
        if not self.read_params_into_session():
            return
        self.session.step()
        self.refresh_all()

    def on_start(self):
        if not self.read_params_into_session():
            return
        self.session.running = True
        self._tick()

    def on_stop(self):
        self.session.running = False

    def _tick(self):
        if not self.session.running:
            return
        if self.session.done():
            self.session.running = False
            self.refresh_all()
            return

        self.session.step()
        self.refresh_all()
        # 60 ms = ok. 16 FPS; możesz zmienić
        self.after(60, self._tick)


if __name__ == "__main__":
    app = GAApp()
    app.mainloop()