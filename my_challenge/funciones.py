# -*- coding: utf-8 -*-
"""
Funciones para main.py
@author: lupev
"""
# Posicion barco 
def pos_barcos_aleatorio():
    pass

# ejemplo posicionamiento aleatorio 
# pos_barco =  np.array([[1, 7]])
# while pos_barco[0][1]+4 >len(tablero):
#     pos_barco =  np.random.randint( 0, 10, size= (1,2))
# print(pos_barco)    

# for fil, vec in enumerate(tablero):
#     for col, val in enumerate(vec):
#         if val != "O" and col+4 <len(tablero):
#             if fil == pos_barco[0][0] and col == pos_barco[0][1]:
#                 tablero[fil:fil+1,col:col+4] = "O"
    

def pos_manual(coordenadas):
    pass

#poner barcos en el tablero 
def poner_barcos():
    pass

#comprobar barco exite
#poner marca X(disparo), O (barco) o - (agua)
def comprobar_barco_existe(tablero, coordenada): 
    pass
    return tablero

def disparo(tablero, barco):
    for pieza in barco:
        tablero[pieza] = "X"
    return tablero


def coloca_barco(tablero, barco):
    for pieza in barco:
        tablero[pieza] = "O"
    return tablero

