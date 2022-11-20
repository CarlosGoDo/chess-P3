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


    def isCheckMate(self, currentState_Player, currentState_Rival):
        # Your Code
        if len(currentState_Rival) == 0 or len(currentState_Player) == 0:
            return True
        if len(currentState_Rival) == 1  and (currentState_Rival[0][2]== 8 or currentState_Rival[0][2]== 2):
            return True
        if len(currentState_Player) == 1 and (currentState_Player[0][2]== 2 or currentState_Player[0][2]== 8):
            return True
        return False

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
        else:# sila funcion
            return -value


    def miniMax(self,currentStatePlayer,currentStateRival,depth, player=True,color= True):

        if depth == 0 or self.isCheckMate(currentStatePlayer, currentStateRival):
            return None, self.evaluate(currentStatePlayer, currentStateRival,color)

        best_move = None
        if player:
            max_eval = -9999999
            lista_player = self.getListnextStatesX(currentStatePlayer)
            for nei in lista_player:
                if self.nei_corrector(nei, currentStatePlayer):
                    chess_temp = copy.deepcopy(self.chess)
                    self.hacer_movimiento(currentStatePlayer, nei)
                    self.elimina_piece(nei, currentStateRival)
                    current_eval = self.miniMax(currentStatePlayer, currentStateRival, depth - 1, False, color)[1]
                    self.chess = chess_temp
                    if color:  # si color es true, minimax ha sido llamado por los blancos.
                        currentStatePlayer = self.chess.boardSim.currentStateW
                        currentStateRival = self.chess.boardSim.currentStateB
                    else:  # si es false ha sido llamado por los negros.
                        currentStatePlayer = self.chess.boardSim.currentStateB
                        currentStateRival = self.chess.boardSim.currentStateW
                    if current_eval >= max_eval:
                        max_eval = current_eval
                        best_move = nei
            return best_move, max_eval

        else:
            min_eval = 9999999
            lista_rival = self.getListnextStatesX(currentStateRival)
            for nei in lista_rival:
                if self.nei_corrector(nei, currentStateRival):
                    chess_temp = copy.deepcopy(self.chess)
                    self.hacer_movimiento(currentStateRival, nei)
                    self.elimina_piece(nei, currentStatePlayer)
                    current_eval = self.miniMax(currentStatePlayer, currentStateRival, depth - 1, True, color)[1]
                    self.chess = chess_temp
                    if color:  # si color es true minimax ha sido llamado por los blancos.
                        currentStatePlayer = self.chess.boardSim.currentStateW
                        currentStateRival = self.chess.boardSim.currentStateB
                    else:  # si es false ha sido llamado por los negros.
                        currentStatePlayer = self.chess.boardSim.currentStateB
                        currentStateRival = self.chess.boardSim.currentStateW
                    if current_eval <= min_eval:
                        min_eval = current_eval
                        best_move = nei
            return best_move, min_eval


    def max_value(self, currentState):
        # Your Code here

        v = float('-inf')
        return 0

    def min_value(self, currentState):
        # Your Code here

        v = float('inf')
        return 0

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
    TA[7][5] = 6



    TA[0][0] = 8
    TA[0][5] = 12
    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentStateB = aichess.chess.boardSim.currentStateB
    currentStateW = aichess.chess.boardSim.currentStateW

    print("printing board")
    aichess.chess.boardSim.print_board()
    temp = copy.deepcopy(aichess.chess)

    # get list of next states for current state
    print("current State Black", currentStateB)
    print("current State White", currentStateW)

    print("siguientes estados",aichess.getListNextStatesW(currentStateW))
    # it uses board to get them... careful 

    #print("list next states ", aichess.listNextStates)

    # starting from current state find the end state (check mate) - recursive function
    # aichess.chess.boardSim.listVisitedStates = []
    # find the shortest path, initial depth 0
    depth = 4
    """
    ches_temp = copy.deepcopy(aichess.chess)
    aux = aichess.miniMax( currentStateW,currentStateB,depth)
    print("siguiente estado ", aux)
    aichess.chess = ches_temp
    currentStateW = aichess.chess.boardSim.currentStateW
    currentStateB = aichess.chess.boardSim.currentStateB
    aichess.hacer_movimiento(currentStateW,[[0, 7, 2], [4, 6, 6]])
    aichess.elimina_piece(currentStateW,currentStateB)
    aichess.chess.boardSim.print_board()
    """
    #aux2 = aichess.miniMax_B(currentStateB,currentStateW,depth)
    #print("el siguiente estado ", aux2)




    i = 1
    check = 1
    while check != 0:

        if i%2 !=0:
            chess_temp = copy.deepcopy(aichess.chess)
            aux = aichess.miniMax(currentStateW, currentStateB, depth, True)[0]
            currentStateW = aichess.chess.boardSim.currentStateW
            currentStateB = aichess.chess.boardSim.currentStateB
            print("El estado encontrado ", aux)
            #time.sleep(10)
            if aux != None:
                aichess.chess = chess_temp
                aichess.hacer_movimiento(currentStateW, aux)
                aichess.elimina_piece(aux, currentStateB)
                #print("Blancas",currentStateW)
                #print("Nergas", currentStateB)
                #print("Blancas board sim", aichess.chess.boardSim.currentStateW)
                print("Negras board sim", aichess.chess.boardSim.currentStateB)
                currentStateW = aichess.chess.boardSim.currentStateW
                currentStateB = aichess.chess.boardSim.currentStateB

                #time.sleep(10)
            else:
                print("soy nulo")
            i += 1
        else:
            chess_temp = copy.deepcopy(aichess.chess)
            aux = aichess.miniMax(currentStateB, currentStateW, depth, True,False)[0]
            currentStateW = aichess.chess.boardSim.currentStateW
            currentStateB = aichess.chess.boardSim.currentStateB
            print("El estado encontrado ", aux)
            #time.sleep(15)
            if aux != None:
                aichess.chess = chess_temp
                aichess.hacer_movimiento(currentStateB, aux)
                aichess.elimina_piece(aux, currentStateW)
                print("Blancas", currentStateW)
                print("Nergas", currentStateB)
                print("Blancas board sim", aichess.chess.boardSim.currentStateW)
                print("Negras board sim", aichess.chess.boardSim.currentStateB)
                currentStateW = aichess.chess.boardSim.currentStateW
                currentStateB = aichess.chess.boardSim.currentStateB
                #time.sleep(10)
            else:
                print("soy nulo")
            i += 1
        aichess.chess.boardSim.print_board()

        if aichess.isCheckMate(currentStateW,currentStateB) or i == 1000:
            check = 0



    #print("siguientes estados Blancas: ", aichess.getListNextStatesW(currentStateW))
    #print("siguientes estados Negras: ", aichess.getListNextStatesW(currentStateB))

    #aichess.BreadthFirstSearch(currentState)
    #aichess.DepthFirstSearch(currentState, depth)

    # MovesToMake = ['1e','2e','2e','3e','3e','4d','4d','3c']

    # for k in range(int(len(MovesToMake)/2)):

    #     print("k: ",k)

    #     print("start: ",MovesToMake[2*k])
    #     print("to: ",MovesToMake[2*k+1])

    #     start = translate(MovesToMake[2*k])
    #     to = translate(MovesToMake[2*k+1])

    #     print("start: ",start)
    #     print("to: ",to)

    #     aichess.chess.moveSim(start, to)
    """
    aichess.chess.boardSim.print_board()
    print("#Move sequence...  ", aichess.pathToTarget)
    print("#Visited sequence...  ", aichess.listVisitedStates)
    print("#Current State White...  ", aichess.chess.board.currentStateW)
    print("#Current State Black...  ", aichess.chess.board.currentStateB)
    """
