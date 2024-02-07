import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

def plot_points(points, color):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    plt.scatter(x, y, color=color)

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
    return region_A, region_B

def train_perceptron(region_A, region_B):
    class1_points = region_A
    class2_points = region_B

    perceptron = Perceptron(input_size=2)
    for _ in range(num_iterations):
        for point in class1_points + class2_points:
            inputs = point
            target = 1 if point in class1_points else 0
            perceptron.train(inputs, target, learning_rate)
    return perceptron

def evaluate_recognition(perceptron, test_points):
    correct_predictions = 0
    for point in test_points:
        inputs = point
        target = 1 if point in region_A else 0
        prediction = perceptron.predict(inputs)
        if prediction == target:
            correct_predictions += 1
    recognition_percentage = (correct_predictions / len(test_points)) * 100
    return recognition_percentage

def change_parameters():
    global learning_rate, num_iterations
    learning_rate = float(input("Ingrese el nuevo coeficiente de aprendizaje: "))
    num_iterations = int(input("Ingrese el nuevo número de iteraciones: "))

def plot_with_decision_boundary(region_A, region_B, perceptron):
    plt.clf()
    plot_points(region_A, color='blue')
    plot_points(region_B, color='red')
    plot_decision_boundary(perceptron)
    plt.xlim(*x_range)
    plt.ylim(*y_range)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Perceptrón para clasificación de puntos en el plano')
    plt.pause(0.01)

# Parámetros
num_points_per_class = 4
x_range = (-10, 10)
y_range = (-10, 10)
learning_rate = 0.1
num_iterations = 100

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
        region_A, region_B = generate_regions()
        with open('region_A.txt', 'w') as f:
            for point in region_A:
                f.write(f"{point[0]}, {point[1]}\n")
        with open('region_B.txt', 'w') as f:
            for point in region_B:
                f.write(f"{point[0]}, {point[1]}\n")
    elif option == "2":
        try:
            with open('region_A.txt', 'r') as f:
                region_A = [tuple(map(float, line.strip().split(','))) for line in f]
            with open('region_B.txt', 'r') as f:
                region_B = [tuple(map(float, line.strip().split(','))) for line in f]
            perceptron = train_perceptron(region_A, region_B)
            plot_with_decision_boundary(region_A, region_B, perceptron)
        except FileNotFoundError:
            print("Primero debe generar las regiones.")
    elif option == "3":
        try:
            with open('region_A.txt', 'r') as f:
                region_A = [tuple(map(float, line.strip().split(','))) for line in f]
            with open('region_B.txt', 'r') as f:
                region_B = [tuple(map(float, line.strip().split(','))) for line in f]
            
            choice = input("¿Desea utilizar el mismo conjunto de entrenamiento para el reconocimiento? (s/n): ")
            if choice.lower() == 's':
                test_points_A = region_A
                test_points_B = region_B
                test_points = test_points_A + test_points_B
            else:
                region_A_test = generate_random_points(num_points_per_class, x_range, y_range)
                region_B_test = generate_random_points(num_points_per_class, x_range, y_range)
                test_points = region_A_test + region_B_test
                
            recognition_percentage = evaluate_recognition(perceptron, test_points)
            print(f"Porcentaje de reconocimiento: {recognition_percentage:.2f}%")
        except NameError:
            print("Primero debe entrenar el perceptrón.")
        except FileNotFoundError:
            print("Primero debe generar las regiones.")
    elif option == "4":
        change_parameters()
    elif option == "5":
        print("¡Hasta luego!")
        break
    else:
        print("Opción inválida. Por favor, seleccione una opción válida.")

plt.show()
