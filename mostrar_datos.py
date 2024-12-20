import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Lee la actividad de los argumentos del script
if len(sys.argv) < 2:
    print("Error: Debes especificar la actividad como argumento.")
    sys.exit(1)  # Sale si no se proporciona la actividad

actividad = sys.argv[1]
nombre_fichero = f"datos_wii_{actividad}.csv"

# Intentar cargar los datos desde el fichero CSV
try:
    # Verificar si el fichero existe
    if not os.path.exists(nombre_fichero):
        print(f"Error: No se encuentra el fichero {nombre_fichero}")
        sys.exit(1)  # Sale si no se encuentra el fichero
    
    # Leer el CSV usando pandas
    data = pd.read_csv(nombre_fichero)
except pd.errors.ParserError:
    print(f"Error: El fichero {nombre_fichero} tiene un formato no válido.")
    sys.exit(1)
except Exception as e:
    print(f"Error inesperado al leer el fichero: {e}")
    sys.exit(1)

# Crear una figura con dos subgráficas
fig, ax = plt.subplots(2, 1)
fig.suptitle(f"Graficas de actividad: {actividad}", fontsize=14)

# Graficar la aceleración en el primer subplot (x, y, z)
ax[0].plot(data.index, data[['x_accel', 'y_accel', 'z_accel']], label=['x', 'y', 'z'])
ax[0].grid(True)
ax[0].legend()

# Graficar la aceleración en el segundo subplot (pitch, roll, yaw)
ax[1].plot(data.index, data[['pitch', 'roll', 'yaw']], label=['pitch', 'roll', 'yaw'])
ax[1].grid(True)
ax[1].legend()

plt.show()
