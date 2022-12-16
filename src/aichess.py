#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy

import chess
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]
import time
from itertools import permutations
import random
import itertools
from random import *


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game

    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.currentStateB = self.chess.boardSim.currentStateB;
        self.depthMax = 8;
        self.checkMate = False
        self.rookValueW = 50;
        self.kingValueW = 900;
        self.rookValueB = -50;
        self.kingValueB = -900;
        self.qTable = np.zeros([8,8,22])

    def hacer_movimiento(self, standard_current_state, standard_next_state):
        start = [e for e in standard_current_state if e not in standard_next_state]
        to = [e for e in standard_next_state if e not in standard_current_state]
        start, to = start[0][0:2], to[0][0:2]
        aichess.chess.moveSim(start, to)


    def nei_corrector(self, nei,estado_actual):
        """
        En esta función observaremos si el nei o estado futuro del tablero al que vamos tiene algún tipo de error
        como poner dos fichas en la misma posición o transformar una torre en rey.
        """
        diferencia = 0
        if (len(nei) > 1):
            if nei[0][2] != nei[1][
                2]:  # En este caso tenemos un estado del tablero futuro donde las 2 fichas són iguales
                if (nei[0][0] != nei[1][0]) or (nei[0][1] != nei[1][
                    1]):  # Aquí comprobamos que las fichas no se superpongan en la misma posición del tablero.
                    if nei[0] != estado_actual[0] and nei[0] != estado_actual[1]:
                        diferencia += 1
                    if nei[1] != estado_actual[0] and nei[1] != estado_actual[1]:
                        diferencia += 1
                    if diferencia == 2:
                        return False
                    return True
        elif len(nei) == 1:
            return True

        return False

    def elimina_piece(self, currentState_move, currentState_not_move):
        """
        Args:
            currentState_move: Jugador que ha realizado el movimiento
            currentState_not_move:Jugador que ¡NO! ha realizado el movimiento.

        Returns:Después de realizar un movimiento se comprueba si existen dos piezas rivales en la misma posición.
        Si se cumple esta condición elimina la pieza de currentState_not_move ya que quiere decir que ha sido eliminada
        del tablero.
        """
        index = 0
        eliminated = None
        for i in range(len(currentState_move)):
            for j in range (len(currentState_not_move)):
                if currentState_move[i][0]==currentState_not_move[j][0] and currentState_move[i][1]==currentState_not_move[j][1]:
                    eliminated = currentState_not_move[j]
                    index = j
        if eliminated != None:
            currentState_not_move.pop(index)
            if eliminated[2] < 7:
                self.chess.boardSim.currentStateW =  currentState_not_move
            else:
                self.chess.boardSim.currentStateB = currentState_not_move

    def getCurrentStateW(self):

        return self.myCurrentStateW

    def getCurrentStateB(self):

        return self.myCurrentStateB

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates
    def getListNextStatesB(self, myState):

        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListnextStatesX(self, mypieces):

        if mypieces[0][2] > 6:
            return self.getListNextStatesB(mypieces)
        else:
            return self.getListNextStatesW(mypieces)


    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    """def isCheckMate(self, currentState_Player, currentState_Rival):
            # Your Code

            if len(currentState_Rival) == 0 or len(currentState_Player) == 0:
                return True
            if len(currentState_Rival) == 1  and (currentState_Rival[0][2]== 8 or currentState_Rival[0][2]== 2):
                return True
            if len(currentState_Player) == 1 and (currentState_Player[0][2]== 2 or currentState_Player[0][2]== 8):
                return True
            return False"""

    def isCheckMate_1(self, mystate):

        # Llista de possibles checkmates
        listCheckMateStates = [[[0, 0, 2], [2, 4, 6]], [[0, 1, 2], [2, 4, 6]], [[0, 2, 2], [2, 4, 6]],
                               [[0, 6, 2], [2, 4, 6]], [[0, 7, 2], [2, 4, 6]],
                               [[2, 4, 6],[0, 0, 2]], [[2, 4, 6],[0, 1, 2]], [[2, 4, 6],[0, 2, 2]],
                               [[2, 4, 6],[0, 6, 2]], [[2, 4, 6],[0, 7, 2]]]

        # Mirem si el nostre estat està a la llista
        if mystate in listCheckMateStates:
            print("is check Mate!")
            return True

        return False


    def isCheckMate_2(self, currentState_Player, currentState_Rival):
        # Your Code

        if len(currentState_Rival) == 0 or len(currentState_Player) == 0:
            return True
        if len(currentState_Rival) == 1  and (currentState_Rival[0][2]== 8 or currentState_Rival[0][2]== 2):
            return True
        if len(currentState_Player) == 1 and (currentState_Player[0][2]== 2 or currentState_Player[0][2]== 8):
            return True
        return False

    def func_heuristic(self, estado1, estado2):
        """
        Calculem la distància de manhattan entre els dos estats passats per paràmetre
        """
        nei1 = copy.copy(estado1)
        nei2 = copy.copy(estado2)

        if nei1[0][2] == 6:
            aux = nei1[0]
            nei1[0] = nei1[1]
            nei1[1] = aux

        if nei2[0][2] == 6:
            aux = nei2[0]
            nei2[0] = nei2[1]
            nei2[1] = aux

        dist1 = abs((nei2[0][0] - nei1[0][0])) + abs((nei2[0][1] - nei1[0][1]))
        dist2 = abs((nei2[1][0] - nei1[1][0])) + abs((nei2[1][1] - nei1[1][1]))

        return dist1 + dist2

    def evaluate(self, currentStatePlayer, currentStateRival, color):
        """

        Args:
            currentStatePlayer: Estado actual del jugador.
            currentStateRival: Estado actual del rival.
            color: Color de las fichas que ha llamado al algoritmo Minimax.

        Returns: Devuelve una evaluación del tablero. Si el algoritmo es llamado por las piezas negras el value será = a
        -value, ya que el valor de las piezas negras es negativo.

        """
        pieces_values = [0, 50, 0, 0, 0, 900,0, -50, 0, 0, 0, -900]
        value = 0
        for piece in currentStatePlayer:
            value += pieces_values[piece[2]-1]

        for piece in currentStateRival:
            value += pieces_values[piece[2]-1]

        if color:
            return value
        else:# si la funcion minimax ha sido llamada por las piezas negras
            return -value
    def reward(self,currentState):

        if self.isCheckMate_2(currentState,self.chess.boardSim.currentStateB):
            return 100

        return -1



    def conversor(self,state):
        """

        Args:
            state: Estado del tablero en forma de string

        Returns:Función auxiliar que transforma un estado del tablero de tipo str a lista de listas.

        """
        mod = state.translate({ord('['): None})
        mod = mod.translate({ord(']'): None})
        mod = mod.split(",")
        mod = [eval(i) for i in mod] #transformamos la list de str a  int.
        chunks = [mod[x:x + 3] for x in range(0, len(mod), 3)]
        return chunks

    def crear_posicion(self,currentState,qlearn,list_moves):
        """

        Args:
            currentState:Estado pasado para la creación.
            qlearn:qlearn con los pesos de los movimientos.

        Returns: Buscamos crear un diccionario para el qlearn donde cada estado tendra, un diccionario de los estados
        futuros a los cuales puede llegar con su puntuación, el reward del propio estado, y la key, que sera el mismo
        CurrentState.

        """
        sta = str(currentState)
        qlearn[sta] = dict()
        qlearn[sta]['value'] = 0
        qlearn[sta]['bestMove'] = None
        qlearn[sta]['moves'] = dict()
        for move in list_moves:
            if qlearn[sta]['moves'].get(str(move)) == None:  # si la key de move no existe qlearn[sta]['moves'][str(move)] = 0
                qlearn[sta]['moves'][str(move)] = 0

        return qlearn[sta]

    def normalize_nei(self, nei):

        if len(nei) > 1 and nei[0][2] > nei[1][2]:
            aux = nei[0]
            nei[0] = nei[1]
            nei[1] = aux
        return nei

    def normalize_list(self, lista, estado_actual):
        i = 0
        for nei in lista:
            if not self.nei_corrector(nei, estado_actual):
                lista.remove(nei)

            else:
                nei = self.normalize_nei(nei)
                lista[i] = nei
            i += 1
        return lista

    def qLearning(self,currentState, num_episodes, discount_factor = 1.0,
                                              epsilon = 0.1, alpha = 0.6):

        qlearn = {}
        lista = self.getListnextStatesX(currentState)
        lista = self.normalize_list(lista,currentState)
        # creamos la primera posición de nuestro Qlearn.
        qlearn[str(currentState)] = self.crear_posicion(currentState,qlearn,lista)
        for iteration in range(num_episodes):
            print("######################################Empieza otra partida", iteration,"#################################")

            state = self.chess.boardSim.currentStateW.copy()
            chess_temp = copy.deepcopy(self.chess)

            print("Estado de las blancas: ", self.chess.boardSim.currentStateW)
            print("Estado de las negras: ", self.chess.boardSim.currentStateB)
            print("state ", state)
            for t in itertools.count():
                list_NextStates = self.getListNextStatesW(state)
                list_NextStates = self.normalize_list(list_NextStates, state)

                if len(list_NextStates) == 0:
                    print("no hay estado sucesores")

                next_state = self.epsilonGreedy(state,list_NextStates,qlearn)
                self.chess.boardSim.print_board()
                print("nos movemos de ",state," ===> ", next_state)
                if state == next_state:
                    print("Atencion tenemos un next_state = state")
                    print(list_NextStates)
                #hacemos el movimiento
                self.hacer_movimiento(state, next_state)
                self.elimina_piece(next_state, self.chess.boardSim.currentStateB)

                print("Estado state: ", state)
                print("Estado next_state: ", next_state)

                #si el siguiente estado es nuevo en el qlearn
                if qlearn.get(str(next_state)) == None:
                    lista = self.getListNextStatesW(next_state)
                    lista = self.normalize_list(lista, next_state)
                    qlearn[str(next_state)] = self.crear_posicion(next_state,qlearn,lista)

                reward = self.reward(next_state)
                qlearn[str(state)]['moves'][str(next_state)] = qlearn[str(next_state)]['value']
                best_next_action = max(qlearn[str(next_state)]['moves'], key=qlearn[str(next_state)]['moves'].get)
                td_target = reward + discount_factor*(qlearn[str(next_state)]['value']) - qlearn[str(state)]['value']
                qlearn[str(state)]['value'] = alpha*td_target
                if self.isCheckMate_2(next_state,self.chess.boardSim.currentStateB):
                    print("Check Mate")
                    self.chess.boardSim.print_board()
                    break
                state = next_state

            self.chess = copy.deepcopy(chess_temp)

        return qlearn

    def epsilonGreedy(self,sta ,listNextStates, qlearn,epsilon = 0.9):
        """

        Args:
            sta: Current state del tablero.
            listNextStates:
            epsilon: Porcentaje de probabilidades de escoger el mejor resultado de q-learn o un movimiento random.

        Returns:Devuelve la siguiente posicion a la que se dirige el bot. 90% de prob que sea el mejor movimiento
        posible, 10% que sea un movimiento random.
        """
        if np.random.rand() < epsilon:
            best_state_str = max(qlearn[str(sta)]['moves'], key=qlearn[str(sta)]['moves'].get)
            best_state = self.conversor(best_state_str)#transformamos el indice del mejor mov de str a lista de listas
            return best_state
        else:
            print()
            x = np.random.randint(0, len(listNextStates))  # Pick a random number between 1 and 100.
            if len(listNextStates) == 0:
                print("no hay estado sucesores")
            return listNextStates[x]

def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """

    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None


if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))
    # white pieces
    # TA[0][0] = 2
    # TA[2][4] = 6
    # # black pieces
    # TA[0][4] = 12

    TA[7][0] = 2
    TA[7][4] = 6

    #TA[0][0] = 8
    TA[0][4] = 12
    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentStateB = aichess.chess.boardSim.currentStateB
    currentStateW = aichess.chess.boardSim.currentStateW

    print("printing board")
    aichess.chess.boardSim.print_board()
    temp = copy.deepcopy(aichess.chess)

    aichess.qLearning(currentStateW,100)

    # get list of next states for current state
    print("current State Black", currentStateB)
    print("current State White", currentStateW)

    print("siguientes estados",aichess.getListNextStatesW(currentStateW))
    print(len(aichess.getListNextStatesW(currentStateW)))
    # it uses board to get them... careful


    aichess.chess.boardSim.print_board()
    print("#Move sequence...  ", aichess.pathToTarget)
    print("#Visited sequence...  ", aichess.listVisitedStates)
    print("#Current State White...  ", aichess.chess.board.currentStateW)
    print("#Current State Black...  ", aichess.chess.board.currentStateB)

