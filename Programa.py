import random
import matplotlib.pyplot as plt
import copy

class Perceptron:
    def __init__(self, tam_entrada, peso_bajo=-1, peso_alto=1):
        # Inicialización de pesos y sesgo (bias) aleatorios
        self.pesos = [random.uniform(peso_bajo, peso_alto) for _ in range(tam_entrada)]
        self.bias = random.uniform(-1, 1)

    # Función de predicción del perceptrón
    def predict(self, entradas):
        activacion = sum(w * x for w, x in zip(self.pesos, entradas)) + self.bias
        return 1 if activacion >= 0 else 0
    
    # Función de entrenamiento del perceptrón
    def entrenamiento(self, entradas, objetivo, coef_aprendizaje):
        prediccion = self.predict(entradas)
        error = objetivo - prediccion
        # Actualización de pesos y sesgo según el error
        self.pesos = [w + coef_aprendizaje * error * x for w, x in zip(self.pesos, entradas)]
        self.bias += coef_aprendizaje * error


# Función para dibujar puntos en un gráfico
def dibuja_puntos(puntos, color, label=None):
    x = [punto[0] for punto in puntos]
    y = [punto[1] for punto in puntos]
    plt.scatter(x, y, color=color, label=label)


# Función para generar regiones A y B con puntos aleatorios
def genera_regiones():
    print("\nIngrese las coordenadas (separadas por un espacio) de los puntos que delimitan la región A:")
    xa, ya = map(float, input("Punto 1 (xa, ya): ").split())
    xb, yb = map(float, input("Punto 2 (xb, yb): ").split())
    
    print("\nIngrese las coordenadas (separadas por un espacio) de los puntos que delimitan la región B:")
    xc, yc = map(float, input("Punto 3 (xc, yc): ").split())
    xd, yd = map(float, input("Punto 4 (xd, yd): ").split())
    
    # Generación de puntos aleatorios dentro de las regiones A y B
    puntos_en_A = [(random.uniform(xa, xb), random.uniform(ya, yb)) for _ in range(puntos_por_clase)]
    puntos_en_B = [(random.uniform(xc, xd), random.uniform(yc, yd)) for _ in range(puntos_por_clase)]
    
    # Guardar los puntos que delimitan las regiones A y B en una lista
    region_A = [(xa, ya), (xb, yb)]
    region_B = [(xc, yc), (xd, yd)]
    
    # Guardar los puntos que delimitan las regiones A y B en archivos
    guarda_archivos(region_A, 'region_A.txt')
    guarda_archivos(region_B, 'region_B.txt')
    
    return puntos_en_A, puntos_en_B


# Función para generar puntos aleatorios dentro de una región
def genera_puntos_aleatorios_en_region(region, num_puntos):
    x = [punto[0] for punto in region]
    y = [punto[1] for punto in region]
    
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    
    return [(random.uniform(x_min, x_max), random.uniform(y_min, y_max)) for _ in range(num_puntos)]


# Función para guardar puntos en un archivo
def guarda_archivos(puntos, nombre_archivo):
    with open(nombre_archivo, 'w') as f:
        for punto in puntos:
            f.write(f"{punto[0]}, {punto[1]}\n")


# Función para cargar puntos desde un archivo
def carga_archivos(nombre_archivo):
    with open(nombre_archivo, 'r') as f:
        puntos = [tuple(map(float, line.strip().split(','))) for line in f]
    return puntos


# Función para entrenar el perceptrón con los puntos de las regiones A y B
def entrenamiento_perceptron(puntos_en_A, puntos_en_B):
    clase1 = puntos_en_A
    clase2 = puntos_en_B

    perceptrons = [Perceptron(tam_entrada=2)]
    for i in range(num_iteraciones):
        for punto in clase1 + clase2:
            entradas = punto
            objetivo = 1 if punto in clase1 else 0
            perceptrons[-1].entrenamiento(entradas, objetivo, coef_aprendizaje)
        # Guardar el estado del perceptrón después de cada iteración
        perceptrons.append(copy.deepcopy(perceptrons[-1]))

    return perceptrons


# Función para dibujar la línea de decisión del perceptrón en el gráfico
def dibuja_linea_decision(perceptron, linestyle, label=None):
    if perceptron.pesos[1] == 0:
        return
    m = -perceptron.pesos[0] / perceptron.pesos[1]      # Pendiente
    b = -perceptron.bias / perceptron.pesos[1]          # Ordenada al origen
    x = [-10, 10]                                       # Dominio
    y = [m * xi + b for xi in x]                        # Ecuación de la recta
    plt.plot(x, y, linestyle=linestyle, color='black', label=label)


# Función para dibujar el proceso de entrenamiento del perceptrón en el gráfico
def dibuja_entrenamiento(puntos_en_A, puntos_en_B, perceptrons, opcion):
    print("\nCierre el gráfico para continuar...")
    plt.clf()
    dibuja_puntos(puntos_en_A, color='blue', label='Clase 1')
    dibuja_puntos(puntos_en_B, color='red', label='Clase 2')
    
    if opcion == "2":  # Opción 2: inicio y final
        dibuja_linea_decision(perceptrons[0], linestyle='--', label='Inicio')
        dibuja_linea_decision(perceptrons[-1], linestyle='-', label='Final')
        plt.legend()
    elif opcion == "1":  # Opción 1: 4 estados
        num_states = 4
        step = len(perceptrons) // num_states
        for i in range(num_states):
            linestyle = '--' if i == 0 else '-' if i == num_states - 1 else '-.'
            dibuja_linea_decision(perceptrons[i * step], linestyle=linestyle, label=f'Estado {i+1}')

        plt.legend()
        
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reconocimiento con un perceptrón')
    plt.show()


# Función para dibujar el reconocimiento con puntos generados aleatoriamente
def dibuja_reconocimiento_aleatorios(puntos_en_A, puntos_en_B, perceptron):
    print("\nCierre el gráfico para continuar...")
    plt.clf()
    dibuja_puntos(puntos_en_A, color='blue', label='Clase 1')
    dibuja_puntos(puntos_en_B, color='red', label='Clase 2')
    dibuja_linea_decision(perceptron, linestyle='-', label='Perceptrón')
    plt.legend()
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reconocimiento con un perceptrón')
    plt.show()


# Función para evaluar el reconocimiento del perceptrón
def evalua_reconocimiento(perceptron, puntos_en_A_prueba, puntos_en_B_prueba):
    puntos_de_prueba = puntos_en_A_prueba + puntos_en_B_prueba

    predicciones_correctas_A = 0
    
    for punto in puntos_de_prueba:
        entradas = punto
        objetivo = 1 if punto in puntos_en_A_prueba else 0
        prediccion = perceptron.predict(entradas)
        if prediccion == objetivo:
            predicciones_correctas_A += 1
    
    a = (predicciones_correctas_A / len(puntos_de_prueba)) * 100

    predicciones_correctas_B = 0
    
    for punto in puntos_de_prueba:
        entradas = punto
        objetivo = 1 if punto in puntos_en_B_prueba else 0
        prediccion = perceptron.predict(entradas)
        if prediccion == objetivo:
            predicciones_correctas_B += 1
    
    b = (predicciones_correctas_B / len(puntos_de_prueba)) * 100

    if(a > b): return a 
    else: return b


# Función para evaluar el reconocimiento ingresando puntos manualmente
def evalua_recon_manual(perceptron, puntos_por_clase):
    puntos_prueba_A = []
    puntos_prueba_B = []
    
    print("\nIngrese las coordenadas de los puntos de la Clase 1:")
    for i in range(puntos_por_clase):
        x, y = map(float, input(f"Ingrese las coordenadas (separadas por un espacio) del punto {i+1} (x, y): ").split())
        puntos_prueba_A.append((x, y))
        
    print("\nIngrese las coordenadas de los puntos de la Clase 2:")
    for i in range(puntos_por_clase):
        x, y = map(float, input(f"Ingrese las coordenadas (separadas por un espacio) del punto {i+1} (x, y): ").split())
        puntos_prueba_B.append((x, y))
    
    porcentaje_de_reconocimiento = evalua_reconocimiento(perceptron, puntos_prueba_A, puntos_prueba_B)

    print(f"\nPorcentaje de reconocimiento: {porcentaje_de_reconocimiento:.2f}%")

    print("\nCierre el gráfico para continuar...")
    
    dibuja_puntos(puntos_prueba_A, color='blue', label='Clase 1')
    dibuja_puntos(puntos_prueba_B, color='red', label='Clase 2')
    dibuja_linea_decision(perceptron, linestyle='-', label='Perceptrón')
    
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Puntos ingresados manualmente con sus clases')
    plt.show()

# Función para cambiar los parámetros del perceptrón
def cambia_parametros():
    global coef_aprendizaje, num_iteraciones, peso_bajo, peso_alto
    coef_aprendizaje = float(input("\nIngrese el nuevo coeficiente de aprendizaje: "))
    num_iteraciones = int(input("Ingrese el nuevo número de iteraciones para el entrenamiento: "))
    peso_bajo = float(input("Ingrese el nuevo límite inferior para la generación de pesos del neurón: "))
    peso_alto = float(input("Ingrese el nuevo límite superior para la generación de pesos del neurón: "))


# Parámetros
puntos_por_clase = 10
x_lim = (-10, 10)
y_lim = (-10, 10)
coef_aprendizaje = 0.001
num_iteraciones = 10
peso_bajo = -1  # Límite inferior para la generación de pesos del neurón
peso_alto = 1   # Límite superior para la generación de pesos del neurón

perceptron = None

# Interfaz de usuario
while True:
    print("\nSeleccione una opción:")
    print("1. Generación de Regiones")
    print("2. Entrenamiento")
    print("3. Reconocimiento")
    print("4. Cambio de Parámetros")
    print("5. Salir")
    opcion = input("\nIngrese el número de la opción que desea ejecutar: ")

    if opcion == "1":
        print("\n¿Desea generar nuevas regiones o utilizar las existentes?")
        print("1. Generar nuevas regiones")
        print("2. Utilizar existentes (si no hay existentes, generará nuevas)")
        region_option = input("\nIngrese el número de la opción: ")
        
        if region_option == "1":
            puntos_por_clase = int(input("\nIngrese cuántos puntos desea generar en cada región (N): "))
            puntos_en_A, puntos_en_B = genera_regiones()

            guarda_archivos(puntos_en_A, 'puntos_en_A.txt')
            guarda_archivos(puntos_en_B, 'puntos_en_B.txt')

        elif region_option == "2":
            try:
                region_A = carga_archivos('region_A.txt')
                region_B = carga_archivos('region_B.txt')
                puntos_por_clase = int(input("\nIngrese cuántos puntos desea generar en cada región (N): "))

                puntos_en_A = genera_puntos_aleatorios_en_region(region_A, puntos_por_clase)
                puntos_en_B = genera_puntos_aleatorios_en_region(region_B, puntos_por_clase)

                guarda_archivos(puntos_en_A, 'puntos_en_A.txt')
                guarda_archivos(puntos_en_B, 'puntos_en_B.txt')
            
            except FileNotFoundError:
                
                print("\nGenerando nuevas regiones...")
                puntos_por_clase = int(input("\nIngrese cuántos puntos desea generar en cada región (N): "))
                puntos_en_A, puntos_en_B = genera_regiones()
                guarda_archivos(puntos_en_A, 'puntos_en_A.txt')
                guarda_archivos(puntos_en_B, 'puntos_en_B.txt')
    
    elif opcion == "2":
        try:
            puntos_en_A = carga_archivos('puntos_en_A.txt')
            puntos_en_B = carga_archivos('puntos_en_B.txt')
            
            perceptron = entrenamiento_perceptron(puntos_en_A, puntos_en_B)
            
            print("\n¿Desea graficar 4 estados del entrenamiento o solo inicio y final?")
            print("1. 4 estados")
            print("2. Inicio y final")
            
            entrenamiento_option = input("\nIngrese el número de la opción: ")
            
            if entrenamiento_option == "1":
                dibuja_entrenamiento(puntos_en_A, puntos_en_B, perceptron, opcion="1")
            elif entrenamiento_option == "2":
                dibuja_entrenamiento(puntos_en_A, puntos_en_B, perceptron, opcion="2")
        
        except FileNotFoundError:
            print("\nPrimero debe generar las regiones.")
    
    elif opcion == "3":
        try:
            region_A = carga_archivos('region_A.txt')
            region_B = carga_archivos('region_B.txt')

            if perceptron is None:  # Verificar si no hay perceptrón entrenado
                print("\nPrimero debe entrenar el perceptrón.")
                continue

            print("\nSeleccione una opción de reconocimiento:")
            print("1. Ingresar N puntos por clase manualmente")
            print("2. Utilizar puntos de entrenamiento")
            print("3. Utilizar N puntos generados aleatoriamente en las regiones")

            recognition_option = input("\nIngrese el número de la opción: ")
            
            if recognition_option == "1":
                puntos_por_clase = int(input("\nIngrese el número de puntos (N) por clase: "))
                evalua_recon_manual(perceptron[-1], puntos_por_clase)
            
            elif recognition_option == "2":
                porcentaje_de_reconocimiento = evalua_reconocimiento(perceptron[-1], puntos_en_A, puntos_en_B)
                print(f"\nPorcentaje de reconocimiento: {porcentaje_de_reconocimiento:.2f}%")
            
            elif recognition_option == "3":
                num_puntos = int(input("\nIngrese el número de puntos (N) por clase: "))
                puntos_en_A_prueba = genera_puntos_aleatorios_en_region(region_A, num_puntos)
                puntos_en_B_prueba = genera_puntos_aleatorios_en_region(region_B, num_puntos)
                
                porcentaje_de_reconocimiento = evalua_reconocimiento(perceptron[-1], puntos_en_A_prueba, puntos_en_B_prueba)
                
                print(f"\nPorcentaje de reconocimiento: {porcentaje_de_reconocimiento:.2f}%")
                dibuja_reconocimiento_aleatorios(puntos_en_A_prueba, puntos_en_B_prueba, perceptron[-1])
        
        except FileNotFoundError:
            print("\nPrimero debe generar las regiones.")

    elif opcion == "4":
        cambia_parametros()
    
    elif opcion == "5":
        print("\n¡Hasta luego!")
        break
    
    else:
        print("\nOpción inválida. Por favor, seleccione una opción válida.")
