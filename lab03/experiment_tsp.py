import sa_tsp
import pso_tsp
import ga_tsp
import time
import statistics
import os
import csv

# Funkcja do uruchamiania eksperymentów
def run_experiment(runs_per_alg=10):
    cities = sa_tsp.load_cities("berlin52.tsp")

    algorithms = {
        "SA": (sa_tsp.simulated_annealing, {'initial_temp':1000, 'cooling_rate':0.995, 'min_temp':1}),
        "PSO": (pso_tsp.pso_tsp, {'num_particles':30, 'max_iter':1000, 'inertia':0.5, 'cognitive':1.5, 'social':1.5}),
        "GA":  (ga_tsp.genetic_algorithm, {'population_size':100, 'generations':500, 'mutation_rate':0.1})
    }

    results = {}

    # prepare results dir
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    for name, (algorithm, params) in algorithms.items():
        print(f"Uruchamianie algorytmu: {name}")
        times = []
        distances = []
        best_paths = []

        for i in range(runs_per_alg):
            start_time = time.time()

            # Wywołanie algorytmu z parametrami
            best_path, best_distance = algorithm(cities, **params)

            end_time = time.time()

            times.append(end_time - start_time)
            distances.append(best_distance)
            best_paths.append(best_path)

        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        avg_distance = statistics.mean(distances)
        std_distance = statistics.stdev(distances) if len(distances) > 1 else 0.0
        best_distance = min(distances)
        worst_distance = max(distances)
        best_path = best_paths[distances.index(best_distance)]

        # zapisz wykres najlepszej trasy do pliku
        img_path = os.path.join(results_dir, f"{name}_best.png")
        save_plot(cities, best_path, img_path)

        results[name] = {
            "average_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "average_distance": avg_distance,
            "std_distance": std_distance,
            "best_distance": best_distance,
            "worst_distance": worst_distance,
            "best_path": best_path,
            "image": img_path
        }

    # zapisz podsumowanie do CSV
    csv_path = os.path.join(results_dir, 'summary.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Algorithm','Best','Average','Std','Worst','AvgTime(s)','MinTime(s)','MaxTime(s)','Image'])
        for name, data in results.items():
            writer.writerow([name, f"{data['best_distance']:.2f}", f"{data['average_distance']:.2f}", f"{data['std_distance']:.2f}", f"{data['worst_distance']:.2f}", f"{data['average_time']:.2f}", f"{data['min_time']:.2f}", f"{data['max_time']:.2f}", data['image']])

    print(f"Wyniki zapisane w: {results_dir}")
    return results


# Funkcja do zapisywania wykresu (obsługuje brak matplotlib)
def save_plot(cities, path, filename):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Brak pakietu 'matplotlib'. Aby zapisać wykresy zainstaluj zależności: pip install -r requirements.txt")
        return
    # Oblicz długość trasy (użycie funkcji z modułu sa_tsp)
    try:
        total_dist = sa_tsp.calculate_distance(cities, path)
    except Exception:
        # fallback: oblicz samodzielnie
        total_dist = 0.0
        for i in range(len(path)):
            c1 = cities[path[i]]
            c2 = cities[path[(i+1) % len(path)]]
            total_dist += ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5

    x = [cities[i][0] for i in path] + [cities[path[0]][0]]
    y = [cities[i][1] for i in path] + [cities[path[0]][1]]
    plt.figure(figsize=(8,6))
    plt.plot(x, y, marker='o')
    # Dodaj opis z długością trasy w lewym górnym rogu
    plt.title('Najlepsza trasa')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax = plt.gca()
    text = f"Długość: {total_dist:.2f}"
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Funkcja do wizualizacji wyników
def visualize_results(results, cities):
    for name, data in results.items():
        print(f"\nAlgorytm: {name}")
        print(f"Średni czas wykonania: {data['average_time']:.2f} s (min: {data['min_time']:.2f}, max: {data['max_time']:.2f})")
        print(f"Średnia długość trasy: {data['average_distance']:.2f}")
        print(f"Odchylenie std.: {data['std_distance']:.2f}")
        print(f"Najlepsza długość trasy: {data['best_distance']:.2f}")
        print(f"Najgorsza długość trasy: {data['worst_distance']:.2f}")
        print(f"Zapisana wizualizacja: {data['image']}")

if __name__ == "__main__":
    results = run_experiment()
    cities = sa_tsp.load_cities("berlin52.tsp")
    visualize_results(results, cities)
