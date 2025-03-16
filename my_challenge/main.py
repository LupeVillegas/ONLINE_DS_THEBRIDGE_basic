# -*- coding: utf-8 -*-
"""
variables usadas en "main.py"
last modification: Mar 16 2025
@authors:LV
"""

from clases import Jugador


#Después de eso ya comienza el juego. 
#Básicamente se irá ejecutando iterativamente en el while,
# y le irá preguntando coordenadas al usuario.

##MEnsaje de bienvenida
print("Mensaje de bienvenida")
vidas = 10

while (vidas !=0):
    
    # Menu para iniciar (llamar a las funciones)
    
    # Opcion 1: id = input("Como te llamas?")
    # Opcion 2: Ver instrucciones del juego
    # Opcion 3: Jugar--->inicializar tableros
    # Opcion 5: Dejar de jugar 
    print("print menu")

    ##Use a switch()
    
    #inicializar tableros
    #maquina debe ingresar barcos aleatoriamente
    #jugador debe ingresar las coordenadas
    jugador1 = Jugador(id_jugador = id)
    
    vidas -=1
    
    


