#!/usr/bin/env python3

from sys import argv, stdout
from threading import Thread
from threading import Condition
import GameData
import socket
from constants import *
from agent import Agent
import os

def main():
    if len(argv) < 4:
        print("You need the player name to start the game.")
        #exit(-1)
        playerName = "Test" # For debug
        ip = HOST
        port = PORT
    else:
        playerName = argv[3]
        ip = argv[1]
        port = int(argv[2])

    players = []

    agent = None

    run = True

    hint_received = False

    statuses = ["Lobby", "Game", "GameHint"]

    status = statuses[0]

    hintState = ("", "")

    def exit_action():
        global run
        run = False
        os._exit(0)

    def show_action():
        if status == statuses[1]:
            s.send(GameData.ClientGetGameStateRequest(playerName).serialize())

    def discard_action(cardOrder):
        try:
            s.send(GameData.ClientPlayerDiscardCardRequest(playerName, cardOrder).serialize())
        except:
            print("Maybe you wanted to type 'discard <num>'?")

    def play_action(cardOrder):
        if status == statuses[1]:
            try:
                s.send(GameData.ClientPlayerPlayCardRequest(playerName, cardOrder).serialize())
            except:
                print("Maybe you wanted to type 'play <num>'?")

    def hint_action(destination, t, value):
        try:
            if t != "colour" and t != "color" and t != "value":
                print("Error: type can be 'color' or 'value'")
            if t == "value":
                value = int(value)
                if int(value) > 5 or int(value) < 1:
                    print("Error: card values can range from 1 to 5")
            else:
                if value not in ["green", "red", "blue", "yellow", "white"]:
                    print("Error: card color can only be green, red, blue, yellow or white")
            s.send(GameData.ClientHintData(playerName, destination, t, value).serialize())
        except:
            print("Maybe you wanted to type 'hint <type> <destinatary> <value>'?")

    def manageInput(cv):
        with cv:
            while run:
                print("waiting on cv")
                cv.wait() # wait for our turn
                print("cv notified!")
                try:
                    s.send(agent.make_move().serialize())
                except Exception as e:
                    print(e)
                stdout.flush()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        request = GameData.ClientPlayerAddData(playerName)
        s.connect((HOST, PORT))
        s.send(request.serialize())
        data = s.recv(DATASIZE)
        data = GameData.GameData.deserialize(data)
        if type(data) is GameData.ServerPlayerConnectionOk:
            print("Connection accepted by the server. Welcome " + playerName)
            s.send(GameData.ClientPlayerStartRequest(playerName).serialize())

        cv = Condition()
        Thread(target=manageInput, args=(cv,)).start()

        while run:
            dataOk = False
            data = s.recv(DATASIZE)
            if not data:
                continue
            data = GameData.GameData.deserialize(data)

            # received when one player send the "ready"
            if type(data) is GameData.ServerPlayerStartRequestAccepted:
                dataOk = True
                print("Ready: " + str(data.acceptedStartRequests) + "/"  + str(data.connectedPlayers) + " players")
                data = s.recv(DATASIZE)
                data = GameData.GameData.deserialize(data)

            # received when all players are ready
            if type(data) is GameData.ServerStartGameData:
                dataOk = True
                print("Game start!")

                players = data.players
                s.send(GameData.ClientPlayerReadyData(playerName).serialize())
                status = statuses[1]
                show_action()


            # received when the command "show" is sent
            if type(data) is GameData.ServerGameStateData:
                dataOk = True

                if agent is None:
                    agent = Agent(playerName, data, players)
                    print(agent.knowledge.to_string())
                else:
                    if (not hint_received):
                        agent.update_knowledge(data.players)
                    else:
                        hint_received = False
                    print(agent.knowledge.to_string())
                    print(agent.hands)
                    print(agent.trash)
                    print(agent.board)
                if (data.currentPlayer == agent.name):
                    print("agent turn")
                    with cv:
                        cv.notify()

            if type(data) is GameData.ServerActionInvalid:
                dataOk = True
                # something is wrong, shouldn't be here
                print("Invalid action performed. Reason:")
                print(data.message)
                stdout.flush()
                run = False

            # received when one player discard a card 
            if type(data) is GameData.ServerActionValid:
                dataOk = True
                print("Action valid!")
                print("Current player: " + data.player)
                agent.update_last_action(data)
                show_action()
                ## TODO
                # check CV

            # received when one player play a card
            if type(data) is GameData.ServerPlayerMoveOk:
                dataOk = True
                print("Nice move!")
                print("Current player: " + data.player)
                agent.update_last_action(data)
                show_action()
                ## TODO
                # update agent state
                # check CV

            # received when one player makes a mistake
            if type(data) is GameData.ServerPlayerThunderStrike:
                dataOk = True
                print("OH NO! The Gods are unhappy with you!")
                agent.update_last_action(data)
                show_action()
                ## TODO
                # update agent state

            # received when one player hint another player
            if type(data) is GameData.ServerHintData:
                dataOk = True
                print("Hint type: " + data.type)
                print("Player " + data.destination + " cards with value " + str(data.value) + " are:")
                for i in data.positions:
                    print("\t" + str(i))
                # agent.update_last_action(data)
                agent.update_knowledge_on_hint_received(data)
                if (data.destination == agent.name):
                    hint_received = True
                    show_action()
                    



            if type(data) is GameData.ServerInvalidDataReceived:
                dataOk = True
                print(data.data)
                ## TODO
                # something went wrong, it shouldn't happen

            if type(data) is GameData.ServerGameOver:
                dataOk = True
                print(data.message)
                print(data.score)
                print(data.scoreMessage)
                stdout.flush()
                run = False
                ## TODO
                # stop client thread (destroy CV)

            if not dataOk:
                print("Unknown or unimplemented data type: " +  str(type(data)))
            # print("[" + playerName + " - " + status + "]: ", end="")
            stdout.flush()


if __name__ == '__main__':
    main()