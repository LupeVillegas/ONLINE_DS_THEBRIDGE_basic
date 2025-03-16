"""
Clases usadas en "main.py"
last modification: Mar 15 2025
@authors:LV
"""
from variables import dim_tablero, num_barcos
from funciones import *

class Jugador:
    """
    
    
    """
    dim_tablero = dim_tablero
    num_barcos = num_barcos
    
    
    #barcos:un diccionario donde nombre de tus
    #tus barcos, y la eslora de cada uno.
    #tablero_vacio:array de numpy donde posicionarás los barcos.
    #Este tablero está vacío, por lo que lo puedes rellenar 
    #de 0s, 1s, o el caracter que consideres.
    #tablero_juego: array de numpy con sus barcos 
    #array con los disparos efectuados, para saber dónde 
    #tenemos que disparar.
    def __init__(self, id_jugador, barcos,tablero_vacio, tablero_juego ):
        self.id = id_jugador
        self.tablero_v = tablero_vacio
        self.tablero_j = tablero_juego
        self.barcos = barcos
        
    #introducir barcos en el tablero 
    def posicionar_barcos(self, id_jugador):
        ###### poner aqui las funciones
        #pos_barcos_aleatorio()
        #pos_manual()
        #poner_barcos()
        #imprimir_tableros()
        pass
    
    #jugador en ese tablero, tendrás que comprobar si ahi
    #había un barco, o simplemente agua. 
    #Acuérdate de marcar en el tablero, tanto si hay un impacto,
    #como si dio agua.
    def disparo_coordenada(self, id_jugador, coordenada):
        self.coordenada = coordenada
        #comprobar_barco_existe()
        #poner_marca()
        