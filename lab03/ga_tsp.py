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

# Algorytm genetyczny dla TSP
def genetic_algorithm(cities, population_size, generations, mutation_rate):
    def crossover(parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end] = parent1[start:end]
        pointer = 0
        for gene in parent2:
            if gene not in child:
                while child[pointer] != -1:
                    pointer += 1
                child[pointer] = gene
        return child

    def mutate(path):
        if random.random() < mutation_rate:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]

    # Inicjalizacja populacji
    population = [random.sample(range(len(cities)), len(cities)) for _ in range(population_size)]
    for generation in range(generations):
        # Ocena populacji
        fitness = [(calculate_distance(cities, individual), individual) for individual in population]
        fitness.sort(key=lambda x: x[0])
        population = [individual[:] for _, individual in fitness[:population_size // 2]]

        # Tworzenie nowej populacji
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)
        population = new_population

    best_individual = min(population, key=lambda x: calculate_distance(cities, x))
    best_distance = calculate_distance(cities, best_individual)
    return best_individual, best_distance

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
    best_path, best_distance = genetic_algorithm(cities, population_size=100, generations=500, mutation_rate=0.1)
    print("Najlepsza długość trasy:", best_distance)
    plot_path(cities, best_path)
