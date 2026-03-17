import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================
# KONFIGURACJA (studenci mogą zmieniać)
# ======================================
RULE = 30  # 0..255 (np. 30, 90, 110)
N = 101  # liczba komórek
STEPS = 150  # liczba kroków czasu
INIT_MODE = "single"  # "single" albo "random"
P_RANDOM = 0.5  # gęstość jedynek dla INIT_MODE="random"
BOUNDARY = "wrap"  # "wrap" (periodyczne) albo "fixed" (zera na brzegach)
SEED = 0  # ziarno losowości


# ======================================
# 1) KONWERSJA REGUŁY NA BITY
# ======================================
def rule_to_bits(rule: int) -> np.ndarray:
    """
    Zamienia regułę 0..255 na tablicę 8 bitów.
    Kolejność powinna odpowiadać konfiguracjom:
    111, 110, 101, 100, 011, 010, 001, 000
    """

    if not (0 <= rule <= 255):
        raise ValueError("rule must be in range 0..255")
    b = np.binary_repr(rule, width=8)
    bits = np.array([int(ch) for ch in b], dtype=np.uint8)
    return bits


# ======================================
# 2) INICJALIZACJA STANU POCZĄTKOWEGO
# ======================================
def init_cells(n: int, mode: str = "single", p: float = 0.5, seed: int = 0) -> np.ndarray:
    """
    Tworzy stan początkowy (wektor długości n).
    mode:
    - 'single': jedna jedynka w środku
    - 'random': losowy wektor 0/1 o gęstości p
    """

    rng = np.random.default_rng(seed)
    cells = np.zeros(n, dtype=np.uint8)
    if mode == "single":
        mid = n // 2
        cells[mid] = 1
    elif mode == "random":
        cells = rng.random(n) < p
        cells = cells.astype(np.uint8)
    else:
        raise ValueError("mode must be 'single' or 'random'")
    return cells


# ======================================
# 3) JEDEN KROK AUTOMATU (ECA)
# ======================================
def step_eca(cells: np.ndarray, rule_bits: np.ndarray, boundary: str = "wrap") -> np.ndarray:
    """
    Wykonuje jeden krok automatu Wolframa.
    boundary:
    - 'wrap': periodyczne (torus)
    - 'fixed': stałe zera poza brzegami
    """
    n = len(cells)
    new = np.zeros(n, dtype=np.uint8)

    # obsługa brzegów
    def get(i: int) -> int:
        if boundary == "wrap":
            return int(cells[i % n])
        elif boundary == "fixed":
            if i < 0 or i >= n:
                return 0
            return int(cells[i])
        else:
            raise ValueError("boundary must be 'wrap' or 'fixed'")

    for i in range(n):
        left = get(i - 1)
        center = get(i)
        right = get(i + 1)
        pattern = (left << 2) | (center << 1) | right
        index = 7 - pattern
        new[i] = rule_bits[index]
    return new

# ======================================
# 4) SYMULACJA
# ======================================
def simulate(rule: int, n: int, steps: int, init_mode: str, p_random: float, boundary: str, seed: int) -> np.ndarray:
    """
    Zwraca historię: macierz (steps+1) x n
    """

    rule_bits = rule_to_bits(rule)
    cells = init_cells(n, mode=init_mode, p=p_random, seed=seed)
    history = np.zeros((steps + 1, n), dtype=np.uint8)
    history[0] = cells
    for t in range(1, steps + 1):
        cells = step_eca(cells, rule_bits, boundary=boundary)
        history[t] = cells
    return history

# ======================================
# 5) METRYKI: gęstość i zmienność
# ======================================
def metrics(history: np.ndarray) -> tuple[float, float]:
    """
    Zwraca:
    - rho: średnia gęstość jedynek
    - V: średnia zmienność (odsetek komórek zmieniających stan między krokami)
    """

    rho = float(history.mean())
    if history.shape[0] < 2:
        V = 0.0
    else:
        diffs = (history[1:] != history[:-1]).mean(axis=1)
        V = float(diffs.mean())
    return rho, V


def entropy_time(history: np.ndarray) -> np.ndarray:
    p = history.mean(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = - (p * np.log2(p) + (1 - p) * np.log2(1 - p))
    h = np.nan_to_num(h)
    return h

# ======================================
# 6) WIZUALIZACJA
# ======================================
def plot_history(history: np.ndarray, rule: int, init_mode: str, boundary: str, savepath: str | None = None):
    plt.figure(figsize=(10, 6))
    plt.imshow(history, cmap="binary", interpolation="nearest", aspect="auto")
    plt.title(f"Automat Wolframa – reguła {rule} | init={init_mode} | brzeg={boundary}")
    plt.xlabel("Komórki")
    plt.ylabel("Czas")
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_entropy(entropy: np.ndarray, rule: int, init_mode: str, savepath: str | None = None):
    plt.figure(figsize=(8, 4))
    plt.plot(entropy, label="Entropia")
    plt.title(f"Entropia w czasie – reguła {rule} | init={init_mode}")
    plt.xlabel("Czas")
    plt.ylabel("Entropia")
    plt.legend()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ======================================
# MAIN
# ======================================
if __name__ == "__main__":
    rules = [30, 90, 110]
    init_modes = ["single", "random"]
    seed = SEED

    outdir = os.path.abspath(os.path.dirname(__file__))

    for rule in rules:
        for mode in init_modes:
            history = simulate(rule=rule, n=N, steps=STEPS, init_mode=mode, p_random=P_RANDOM, boundary=BOUNDARY, seed=seed)
            rho, V = metrics(history)
            h = entropy_time(history)
            print(f"Reguła {rule} | init={mode} | rho={rho:.4f} | V={V:.4f} | H_last={h[-1]:.4f}")
            fname = os.path.join(outdir, f"rule_{rule}_init_{mode}.png")
            plot_history(history, rule, mode, BOUNDARY, savepath=fname)
            efname = os.path.join(outdir, f"entropy_rule_{rule}_init_{mode}.png")
            plot_entropy(h, rule, mode, savepath=efname)

    print(f"Wykresy i obrazy entropii zapisane w katalogu: {outdir}")
