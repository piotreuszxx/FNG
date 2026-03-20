import math
import random

# Wczytanie danych z pliku berlin52.tsp
def load_cities(file_path):
    cities = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        start = False
        for line in lines:
            if line.strip() == 'NODE_COORD_SECTION':
                start = True
                continue
            if start:
                parts = line.strip().split()
                if len(parts) < 3 or parts[0] == 'EOF':
                    break
                cities.append((float(parts[1]), float(parts[2])))
    return cities

# Funkcja obliczająca długość trasy
def calculate_distance(cities, path):
    distance = 0
    for i in range(len(path)):
        city1 = cities[path[i]]
        city2 = cities[path[(i + 1) % len(path)]]
        distance += math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
    return distance

# Algorytm PSO dla TSP
def pso_tsp(cities, num_particles, max_iter, inertia, cognitive, social):
    num_cities = len(cities)

    # Inicjalizacja cząstek
    particles = [random.sample(range(num_cities), num_cities) for _ in range(num_particles)]
    velocities = [random.uniform(-1, 1) for _ in range(num_particles)]
    personal_best = [p[:] for p in particles]
    personal_best_distances = [calculate_distance(cities, p) for p in particles]

    global_best = min(personal_best, key=lambda p: calculate_distance(cities, p))
    global_best_distance = calculate_distance(cities, global_best)

    for _ in range(max_iter):
        for i in range(num_particles):
            # Prosty operator ruchu oparty na swapach (adaptacja PSO do permutacji)
            # losowo wykonujemy kilka swapów zależnych od parametrów
            swaps = 1
            if random.random() < 0.1:
                swaps = 2
            for _s in range(swaps):
                idx1, idx2 = random.sample(range(num_cities), 2)
                particles[i][idx1], particles[i][idx2] = particles[i][idx2], particles[i][idx1]

            # Obliczanie nowej odległości
            current_distance = calculate_distance(cities, particles[i])

            # Aktualizacja najlepszego osobistego rozwiązania
            if current_distance < personal_best_distances[i]:
                personal_best[i] = particles[i][:]
                personal_best_distances[i] = current_distance

            # Aktualizacja najlepszego globalnego rozwiązania
            if current_distance < global_best_distance:
                global_best = particles[i][:]
                global_best_distance = current_distance

    return global_best, global_best_distance

# Wizualizacja trasy
def plot_path(cities, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Brak pakietu 'matplotlib'. Aby zobaczyć wykres zainstaluj zależności: pip install -r requirements.txt")
        return
    x = [cities[i][0] for i in path] + [cities[path[0]][0]]
    y = [cities[i][1] for i in path] + [cities[path[0]][1]]
    plt.plot(x, y, marker='o')
    plt.title("Najlepsza trasa")
    plt.show()

if __name__ == "__main__":
    cities = load_cities("berlin52.tsp")
    best_path, best_distance = pso_tsp(cities, num_particles=30, max_iter=1000, inertia=0.5, cognitive=1.5, social=1.5)
    print("Najlepsza długość trasy:", best_distance)
    plot_path(cities, best_path)
