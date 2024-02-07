import random
import matplotlib.pyplot as plt
import copy

class Perceptron:
    def __init__(self, input_size):
        self.weights = [random.uniform(-1, 1) for _ in range(input_size)]
        self.bias = random.uniform(-1, 1)

    def predict(self, inputs):
        activation = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return 1 if activation >= 0 else 0

    def train(self, inputs, target, learning_rate=0.1):
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights = [w + learning_rate * error * x for w, x in zip(self.weights, inputs)]
        self.bias += learning_rate * error

def generate_random_points(num_points, x_range, y_range):
    return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(num_points)]

def plot_points(points, color, label=None):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y, color=color, label=label)

def plot_decision_boundary(perceptron):
    if perceptron.weights[1] == 0:
        return
    slope = -perceptron.weights[0] / perceptron.weights[1]
    intercept = -perceptron.bias / perceptron.weights[1]
    x = [-10, 10]
    y = [slope * xi + intercept for xi in x]
    plt.plot(x, y, color='black')

def generate_regions():
    print("Ingrese las coordenadas de los puntos que delimitan la región A:")
    xa, ya = map(float, input("Punto 1 (xa, ya): ").split())
    xb, yb = map(float, input("Punto 2 (xb, yb): ").split())
    print("Ingrese las coordenadas de los puntos que delimitan la región B:")
    xc, yc = map(float, input("Punto 3 (xc, yc): ").split())
    xd, yd = map(float, input("Punto 4 (xd, yd): ").split())
    region_A = [(random.uniform(xa, xb), random.uniform(ya, yb)) for _ in range(num_points_per_class)]
    region_B = [(random.uniform(xc, xd), random.uniform(yc, yd)) for _ in range(num_points_per_class)]
    # Guardar los puntos que delimitan las regiones A y B en una lista
    points_A = [(xa, ya), (xb, yb)]
    points_B = [(xc, yc), (xd, yd)]
    save_points_to_file(points_A, 'points_A.txt')
    save_points_to_file(points_B, 'points_B.txt')
    return region_A, region_B

def generate_random_points_in_region(region, num_points):
    x_values = [point[0] for point in region]
    y_values = [point[1] for point in region]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    return [(random.uniform(x_min, x_max), random.uniform(y_min, y_max)) for _ in range(num_points)]

def save_points_to_file(points, filename):
    with open(filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]}, {point[1]}\n")

def load_points_from_file(filename):
    with open(filename, 'r') as f:
        points = [tuple(map(float, line.strip().split(','))) for line in f]
    return points

def train_perceptron(region_A, region_B):
    class1_points = region_A
    class2_points = region_B

    perceptrons = [Perceptron(input_size=2)]
    for i in range(num_iterations):
        for point in class1_points + class2_points:
            inputs = point
            target = 1 if point in class1_points else 0
            perceptrons[-1].train(inputs, target, learning_rate)
        # Guardar el estado del perceptrón después de cada iteración
        perceptrons.append(copy.deepcopy(perceptrons[-1]))

    return perceptrons

def plot_decision_boundary(perceptron, linestyle, label=None):
    if perceptron.weights[1] == 0:
        return
    slope = -perceptron.weights[0] / perceptron.weights[1]
    intercept = -perceptron.bias / perceptron.weights[1]
    x = [-10, 10]
    y = [slope * xi + intercept for xi in x]
    plt.plot(x, y, linestyle=linestyle, color='black', label=label)

def plot_with_decision_boundary(region_A, region_B, perceptrons, option):
    plt.clf()
    plot_points(region_A, color='blue', label='Clase 1')
    plot_points(region_B, color='red', label='Clase 2')
    
    if option == "2":  # Opción 2: inicio y final
        plot_decision_boundary(perceptrons[0], linestyle='--', label='Inicio')
        plot_decision_boundary(perceptrons[-1], linestyle='-', label='Final')
        plt.legend()
    elif option == "1":  # Opción 1: 4 estados
        num_states = 4
        step = len(perceptrons) // num_states
        for i in range(num_states):
            linestyle = '--' if i == 0 else '-' if i == num_states - 1 else '-.'
            plot_decision_boundary(perceptrons[i * step], linestyle=linestyle, label=f'Estado {i+1}')

        plt.legend()
        
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Perceptrón para clasificación de puntos en el plano')
    plt.show()

def plot_with_decision_boundary_and_points(region_A, region_B, perceptron):
    plt.clf()
    plot_points(region_A, color='blue', label='Clase 1')
    plot_points(region_B, color='red', label='Clase 2')
    plot_decision_boundary(perceptron, linestyle='-', label='Perceptrón')
    plt.legend()
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Perceptrón para clasificación de puntos en el plano')
    plt.show()

def evaluate_recognition(perceptron, test_points, region_A_test):
    correct_predictions = 0
    for point in test_points:
        inputs = point
        target = 1 if point in region_A_test else 0
        prediction = perceptron.predict(inputs)
        if prediction == target:
            correct_predictions += 1
    recognition_percentage = (correct_predictions / len(test_points)) * 100
    return recognition_percentage

def change_parameters():
    global learning_rate, num_iterations
    learning_rate = float(input("Ingrese el nuevo coeficiente de aprendizaje: "))
    num_iterations = int(input("Ingrese el nuevo número de iteraciones: "))


def evaluate_manual_recognition(perceptron, num_points_per_class):
    if perceptron is None:
        print("Primero debe entrenar el perceptrón.")
        return

    test_points_A = []
    test_points_B = []
    
    print("Ingrese las coordenadas de los puntos de la Clase 1:")
    for i in range(num_points_per_class):
        x, y = map(float, input(f"Ingrese las coordenadas del punto {i+1} (x, y): ").split())
        test_points_A.append((x, y))
        
    print("Ingrese las coordenadas de los puntos de la Clase 2:")
    for i in range(num_points_per_class):
        x, y = map(float, input(f"Ingrese las coordenadas del punto {i+1} (x, y): ").split())
        test_points_B.append((x, y))

    correct_predictions = 0
    total_points = num_points_per_class * 2
    
    # Evaluación de todos los puntos
    for point in test_points_A + test_points_B:
        inputs = point
        prediction = perceptron.predict(inputs)
        # Si la predicción es correcta, incrementamos correct_predictions
        if (point in test_points_A and prediction == 1) or (point in test_points_B and prediction == 0):
            correct_predictions += 1

    recognition_percentage = (correct_predictions / total_points) * 100
    print(f"Porcentaje de reconocimiento: {recognition_percentage:.2f}%")
    
    # Plot de los puntos ingresados manualmente con sus clases
    plot_points(test_points_A, color='blue', label='Clase 1')
    plot_points(test_points_B, color='red', label='Clase 2')
    plot_decision_boundary(perceptron, linestyle='-', label='Perceptrón')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Puntos ingresados manualmente con sus clases')
    plt.show()


# Parámetros
num_points_per_class = 10
x_range = (-10, 10)
y_range = (-10, 10)
learning_rate = 0.1
num_iterations = 100

perceptron = None

# Interfaz de usuario
while True:
    print("\nSeleccione una opción:")
    print("1. Generación de Regiones")
    print("2. Entrenamiento")
    print("3. Reconocimiento")
    print("4. Cambio de Parámetros")
    print("5. Salir")
    option = input("Ingrese el número de la opción que desea ejecutar: ")

    if option == "1":
        num_points = int(input("Ingrese cuántos puntos desea generar en cada región (N): "))
        print("¿Desea generar nuevas regiones o utilizar las existentes?")
        print("1. Generar nuevas regiones")
        print("2. Utilizar existentes (si no hay existentes, generará nuevas)")
        region_option = input("Ingrese el número de la opción: ")
        if region_option == "1":
            region_A, region_B = generate_regions()
            save_points_to_file(region_A, 'region_A.txt')
            save_points_to_file(region_B, 'region_B.txt')
        elif region_option == "2":
            try:
                region_A = load_points_from_file('region_A.txt')
                region_B = load_points_from_file('region_B.txt')
            except FileNotFoundError:
                print("Generando nuevas regiones...")
                region_A, region_B = generate_regions()
                save_points_to_file(region_A, 'region_A.txt')
                save_points_to_file(region_B, 'region_B.txt')
    elif option == "2":
        try:
            region_A = load_points_from_file('region_A.txt')
            region_B = load_points_from_file('region_B.txt')
            perceptron = train_perceptron(region_A, region_B)
            print("¿Desea graficar 4 estados del entrenamiento o solo inicio y final?")
            print("1. 4 estados")
            print("2. Inicio y final")
            train_option = input("Ingrese el número de la opción: ")
            if train_option == "1":
                plot_with_decision_boundary(region_A, region_B, perceptron, option="1")
            elif train_option == "2":
                plot_with_decision_boundary(region_A, region_B, perceptron, option="2")
        except FileNotFoundError:
            print("Primero debe generar las regiones.")
    elif option == "3":
        try:
            points_A = load_points_from_file('points_A.txt')
            points_B = load_points_from_file('points_B.txt')

            if perceptron is None:  # Verificar si no hay perceptrón entrenado
                print("Primero debe entrenar el perceptrón.")
                continue

            print("Seleccione una opción de reconocimiento:")
            print("1. Ingresar N puntos manualmente")
            print("2. Utilizar puntos de entrenamiento")
            print("3. Utilizar N puntos generados aleatoriamente en las regiones")

            recognition_option = input("Ingrese el número de la opción: ")
            if recognition_option == "1":
                num_points_per_class = int(input("Ingrese el número de puntos (N) por clase: "))
                evaluate_manual_recognition(perceptron[-1], num_points_per_class)
            elif recognition_option == "2":
                test_points_A = region_A
                test_points_B = region_B
                test_points = test_points_A + test_points_B
                recognition_percentage = evaluate_recognition(perceptron[-1], test_points, region_A)
                print(f"Porcentaje de reconocimiento: {recognition_percentage:.2f}%")
            elif recognition_option == "3":
                num_points = int(input("Ingrese el número de puntos (N) por clase: "))
                region_A_test = generate_random_points_in_region(points_A, num_points)
                region_B_test = generate_random_points_in_region(points_B, num_points)
                test_points = region_A_test + region_B_test
                recognition_percentage = evaluate_recognition(perceptron[-1], test_points, region_A_test)
                print(f"Porcentaje de reconocimiento: {recognition_percentage:.2f}%")
                plot_with_decision_boundary_and_points(region_A_test, region_B_test, perceptron[-1])
        except FileNotFoundError:
            print("Primero debe generar las regiones.")

    elif option == "4":
        change_parameters()
    elif option == "5":
        print("¡Hasta luego!")
        break
    else:
        print("Opción inválida. Por favor, seleccione una opción válida.")
