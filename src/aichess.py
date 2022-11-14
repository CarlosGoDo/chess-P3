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
    def nei_corrector(self, nei):
        """
        En esta función observaremos si el nei o estado futuro del tablero al que vamos tiene algún tipo de error
        como poner dos fichas en la misma posición o transformar una torre en rey.
        """
        if(len(nei)>1):
            if nei[0][2] != nei[1][2]: #En este caso tenemos un estado del tablero futuro donde las 2 fichas són iguales
                if (nei[0][0] != nei[1][0]) and (nei[0][1] != nei[1][1]):#Aquí comprobamos que las fichas no se superpongan en la misma posición del tablero.
                    return True
        elif len(nei)== 1:
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
                currentState_not_move = self.chess.boardSim.currentStateW
            else:
                currentState_not_move = self.chess.boardSim.currentStateB

        return  currentState_not_move

    def getCurrentState(self):

        return self.myCurrentStateW

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


    def isCheckMate(self, currentStateW, currentStateB):

        # Your Code

        if len(currentStateB) == 0 or len(currentStateW) == 0:
            return True
        if len(currentStateB) == 1  and currentStateB[0][2]== 8:
            return True
        if len(currentStateW) == 1 and currentStateW[0][2]== 2:
            return True
        return False

    def evaluate(self, currentStateW, currentStateB):

        value = 0
        if len(currentStateB) == 2:
            #print("entro en len 2 B ")
            value += (self.rookValueB + self.kingValueB)

        if len(currentStateW) == 2:
            #print("entro en len 2 W")
            value += (self.rookValueW + self.kingValueW)

        if len(currentStateW) == 1:
            if currentStateW[0][2] == 2:
                #print("entro en len 1 W R ")
                value += self.rookValueW
            elif currentStateW[0][2] == 6:
                #print("entro en len 1 W K ")
                value += self.kingValueW

        if len(currentStateB) == 1:
            if currentStateB[0][2] == 8:
                #print("entro en len 1 B R ")
                value += self.rookValueB
            elif currentStateB[0][2] == 12:
                #print("entro en len 1 B K ")
                value += self.kingValueB
        return value

    def miniMax(self,currentStatePlayer,currentStateRival,depth, player):
        # Your Code here
        if depth == 0 or self.isCheckMate(currentStatePlayer, currentStateRival):
            lista = self.getListnextStatesX(currentStatePlayer)
            #print("siguientes estados estamos en el check: ", lista)
            metrica = self.evaluate(currentStatePlayer, currentStateRival)
            #print("valor de la jugada: ",metrica)
            return metrica, currentStatePlayer
        if player:
            maxEval = float('-inf')
            best_move = None
            lista_player = self.getListnextStatesX(currentStatePlayer)
            for nei in lista_player:
                if self.nei_corrector(nei):
                    chess_temp = copy.deepcopy(self.chess)
                    rival_temp = currentStateRival
                    self.hacer_movimiento(currentStatePlayer, nei)
                    currentStateRival = aichess.elimina_piece(nei,currentStateRival)
                    eval = self.miniMax(currentStateRival,nei,depth-1,False)[0]
                    maxEval = max(maxEval,eval)
                    if maxEval == eval:
                        best_move = nei
                    self.chess = chess_temp
                    currentStateRival = rival_temp
            return maxEval, best_move
        else:
            minEval = float('inf')
            best_move = None
            lista_rival = self.getListnextStatesX(currentStateRival)
            #print("siguientes estados: ",lista)
            for nei in lista_rival:
                if self.nei_corrector(nei):
                    temp = copy.deepcopy(self.chess)
                    player_temp = currentStatePlayer
                    self.hacer_movimiento(currentStateRival, nei)
                    currentStatePlayer = aichess.elimina_piece(nei, currentStatePlayer)
                    eval = self.miniMax(currentStatePlayer,nei,depth-1,True)[0]
                    minEval = min(minEval, eval)
                    if minEval == eval:
                        best_move = nei
                    self.chess = temp
                    currentStatePlayer = player_temp
            return minEval, best_move

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
    TA[7][4] = 6


    TA[0][0] = 8
    TA[0][4] = 12

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentStateB = aichess.chess.board.currentStateB.copy()
    currentStateW = aichess.chess.board.currentStateW.copy()

    print("printing board")
    aichess.chess.boardSim.print_board()
    temp = copy.deepcopy(aichess.chess)

    # get list of next states for current state
    print("current State", currentStateB)

    # it uses board to get them... careful 


    #print("list next states ", aichess.listNextStates)

    # starting from current state find the end state (check mate) - recursive function
    # aichess.chess.boardSim.listVisitedStates = []
    # find the shortest path, initial depth 0
    depth = 1
    aichess.chess.boardSim.print_board()
    #aux = aichess.miniMax(currentStateW,currentStateB,depth,True)
    #estados = aichess.getListNextStatesW([[7, 4, 2], [7, 4, 6]])
    #aux = aichess.miniMax(currentStateW,  currentStateB, depth, True)[1]
    #print("siguiente movimiento W", aux)
    #aichess.hacer_movimiento(currentStateW, aux)
    #aichess.elimina_piece(aichess.chess.boardSim.currentStateB, aichess.chess.boardSim.currentStateW)
    #aichess.chess.boardSim.print_board()
    #print("next states", aichess.getListNextStatesW(aichess.chess.boardSim.currentStateW))
    #print(" states", aichess.chess.boardSim.currentStateW)

    #aux2 = aichess.miniMax(currentStateB, currentStateW, depth, True)[1]
    #print("siguientes estados: ",estados)


    #print("siguiente movimiento B", aux2)
    #aichess.chess.boardSim.print_board()


    aichess.hacer_movimiento(currentStateB, [[7, 0, 8], [0, 4, 12]])
    aichess.elimina_piece(aichess.chess.boardSim.currentStateB,aichess.chess.boardSim.currentStateW)
    print(currentStateB,currentStateW)
    print("estados pieza blanca",aichess.chess.boardSim.currentStateW)
    #aichess.chess.boardSim.print_board()
    #print(aichess.chess.currentStateW)
    #aichess.chess.boardSim.print_board()
    #aux1 = [[7, 0, 2], [6, 4, 6]]
    #aichess.hacer_movimiento(currentStateB, aux1)

    """
    i = 1
    tablas = []
    while i !=8:
        tabla = aichess.chess.boardSim
        tablas.append(tabla)
        if i%2 !=0:
            aux = aichess.miniMax(currentStateW, currentStateB, depth, True)[1]
            if aux != None:
                aichess.hacer_movimiento(currentStateW, aux)
                tabla = aichess.chess.boardSim
                tablas.append(tabla)
            else:
                print("soy nulo")
        else:
            aux = aichess.miniMax(currentStateB, currentStateW, depth, True)[1]
            if aux != None:
                aichess.hacer_movimiento(currentStateB, aux)
                tabla = aichess.chess.boardSim
                tablas.append(tabla)
            else:
                print("soy nulo")

        i+=1
        print(i)

    for i in range(len(tablas)):
        tablas[i].print_board()
    """
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

    # aichess.chess.boardSim.print_board()
    print("#Move sequence...  ", aichess.pathToTarget)
    print("#Visited sequence...  ", aichess.listVisitedStates)
    print("#Current State...  ", aichess.chess.board.currentStateW)
