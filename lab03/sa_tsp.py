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

# Funkcja generująca sąsiada (swap dwóch miast)
def generate_neighbor(path):
    new_path = path[:]
    i, j = random.sample(range(len(path)), 2)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path

# Algorytm wyżarzania symulowanego
def simulated_annealing(cities, initial_temp, cooling_rate, min_temp):
    current_path = list(range(len(cities)))
    random.shuffle(current_path)
    current_distance = calculate_distance(cities, current_path)
    best_path = current_path[:]
    best_distance = current_distance

    temperature = initial_temp

    while temperature > min_temp:
        new_path = generate_neighbor(current_path)
        new_distance = calculate_distance(cities, new_path)
        delta = new_distance - current_distance

        if delta < 0 or random.random() < math.exp(-delta / temperature):
            current_path = new_path
            current_distance = new_distance

            if current_distance < best_distance:
                best_path = current_path[:]
                best_distance = current_distance

        temperature *= cooling_rate

    return best_path, best_distance

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
    best_path, best_distance = simulated_annealing(cities, initial_temp=1000, cooling_rate=0.995, min_temp=1)
    print("Najlepsza długość trasy:", best_distance)
    plot_path(cities, best_path)
