from sense_hat import SenseHat
import numpy as np
import time
import os

# Configuración de duración y número de muestras deseadas
duracion = 5.0  # Duración minima en segundos
num_datos = 165  # Número minimo de muestras requeridas

sense = SenseHat()
datos = []

# Pedir al usuario que introduzca la actividad
actividad = input("Introduce el nombre de la actividad que se va a realizar: ")

# Nombre del fichero donde se guardarán los datos
nombre_fichero = f"datos_wii_{actividad}.csv"

# Cuenta regresiva antes de grabar
print("Preparándose para grabar datos...")
for i in range(3, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("¡GRABANDO!")

# Comprobar si el archivo ya existe
if not os.path.exists(nombre_fichero):
    # Si no existe, añadir cabecera al fichero
    with open(nombre_fichero, 'w') as f:
        # Escribir la cabecera
        f.write('actividad,t,pitch,roll,yaw,x_accel,y_accel,z_accel\n')

# Registrar el tiempo de inicio
inicio = time.time()

while time.time() - inicio <= duracion*1.1:
    # Obtener datos del sensor
    ori = sense.get_orientation()
    ace = sense.get_accelerometer_raw()
    
    # Calcular el tiempo actual desde el inicio
    t = time.time() - inicio
    
    # Cada fila se compone de:
    # [1 COL (actividad) +
    #  1 COL (t) +
    #  3 COLS (orientación) +
    #  3 COLS (aceleración)]
    datos_fila = [
        actividad, t, ori['pitch'], ori['roll'], ori['yaw'], 
        ace['x'], ace['y'], ace['z']
    ]
    
    # Agregar datos a la lista
    datos.append(datos_fila)

# Validar el número de muestras
num_capturado = len(datos)

if num_capturado >= num_datos:
    # Truncar datos si hay más de los necesarios
    datos = datos[:num_datos]

    with open(nombre_fichero, 'a') as f:  # Usar modo 'a' para añadir al final del fichero
        # Escribir cada fila en el fichero
        for fila in datos:
            f.write(','.join(map(str, fila)) + '\n')

    # Mensajes finales
    print("Grabación completada.")
    print(f"Datos añadidos al fichero: {nombre_fichero}")
else:
    print(f"Error: No se han tomado suficientes datos ({num_capturado} muestras, mínimo requerido: {num_datos}).")
