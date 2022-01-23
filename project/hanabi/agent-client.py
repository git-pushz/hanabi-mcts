#!/usr/bin/env python3
import sys
from sys import argv, stdout
from threading import Thread
from threading import Condition
import GameData
import socket
import traceback
from constants import *
from agent import Agent, DEBUG, VERBOSE
import os
import traceback
import numpy as np


def main():
    if len(argv) < 4:
        print("You need the player name to start the game.")
        # exit(-1)
        agent_name = "Test"  # For debug
        ip = HOST
        port = PORT
    else:
        agent_name = argv[3]
        ip = argv[1]
        port = int(argv[2])

    players = []
    agent = None
    run = True
    hint_received = False
    statuses = ["Lobby", "Game", "GameHint"]
    status = statuses[0]

    def exit_action():
        nonlocal run
        run = False
        os._exit(0)

    def show_action():
        if status == statuses[1]:
            s.send(GameData.ClientGetGameStateRequest(agent_name).serialize())

    def discard_action(card_order):
        try:
            s.send(GameData.ClientPlayerDiscardCardRequest(agent_name, card_order).serialize())
        except:
            print("Maybe you wanted to type 'discard <num>'?")

    def play_action(card_order):
        if status == statuses[1]:
            try:
                s.send(GameData.ClientPlayerPlayCardRequest(agent_name, card_order).serialize())
            except:
                print("Maybe you wanted to type 'play <num>'?")

    def hint_action(destination, t, value):
        try:
            if t != "colour" and t != "color" and t != "value":
                if DEBUG:
                    print("Error: type can be 'color' or 'value'")
            if t == "value":
                value = int(value)
                if int(value) > 5 or int(value) < 1:
                    if DEBUG:
                        print("Error: card values can range from 1 to 5")
            else:
                if value not in ["green", "red", "blue", "yellow", "white"]:
                    if DEBUG:
                        print("Error: card color can only be green, red, blue, yellow or white")
            s.send(GameData.ClientHintData(agent_name, destination, t, value).serialize())
        except:
            print("Maybe you wanted to type 'hint <type> <destinatary> <value>'?")

    def agent_move_thread():
        with cv:
            while run:
                if DEBUG or VERBOSE:
                    print("waiting on cv")
                cv.wait()  # wait for our turn
                if not run:
                    break
                if DEBUG or VERBOSE:
                    print("cv notified!")
                try:
                    move = agent.make_move()
                    if move is not None and VERBOSE:
                        print(f"I chose the move {move.action}:")
                        if hasattr(move, 'handCardOrdered'):
                            print(f"\tCard: {move.handCardOrdered}")
                        if hasattr(move, 'type') and hasattr(move, 'value'):
                            print(f"\tHint: {move.type} {move.value} to {move.destination}")
                    elif move is None:
                        print("MOVE IS NONE")
                    s.send(move.serialize())
                except Exception:
                    print(traceback.format_exc())
                stdout.flush()

    def check_agent_turn(current_player: str):
        if current_player == agent_name:
            with cv:
                cv.notify()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        request = GameData.ClientPlayerAddData(agent_name)
        s.connect((ip, port))
        s.send(request.serialize())
        data = s.recv(DATASIZE)
        data = GameData.GameData.deserialize(data)
        if type(data) is GameData.ServerPlayerConnectionOk:
            if DEBUG:
                print("Connection accepted by the server. Welcome " + agent_name)
            s.send(GameData.ClientPlayerStartRequest(agent_name).serialize())

        cv = Condition()
        Thread(target=agent_move_thread).start()
        while run:
            dataOk = False
            data = s.recv(DATASIZE)
            if not data:
                continue
            data = GameData.GameData.deserialize(data)

            # 1 received when one player send the "ready"
            if type(data) is GameData.ServerPlayerStartRequestAccepted:
                dataOk = True
                if DEBUG:
                    print("Ready: " + str(data.acceptedStartRequests) + "/" + str(data.connectedPlayers) + " players")
                # data = s.recv(DATASIZE)
                # data = GameData.GameData.deserialize(data)

            # 2 received when all players are ready
            if type(data) is GameData.ServerStartGameData:
                dataOk = True
                print("Game start!")

                players = data.players
                s.send(GameData.ClientPlayerReadyData(agent_name).serialize())
                status = statuses[1]
                # first call -> will initialize the agent and its structures
                show_action()

            # 3 received when the command "show" is sent
            if type(data) is GameData.ServerGameStateData:
                dataOk = True

                if agent is None:
                    agent = Agent(agent_name, data, players)
                    if DEBUG:
                        print(agent.knowledge.to_string())
                else:
                    agent.track_drawn_card(data.players)
                agent.assert_aligned_with_server(data.usedNoteTokens, data.usedStormTokens,
                                                 data.tableCards, data.discardPile, data.players)
                check_agent_turn(data.currentPlayer)

            # 4 received when someone performs an invalid action
            if type(data) is GameData.ServerActionInvalid:
                dataOk = True
                # something is wrong, shouldn't be here
                print("Invalid action performed. Reason:")
                print(data.message)
                stdout.flush()
                run = False
                with cv:
                    cv.notify()

            # 5 received when one player discards a card
            if type(data) is GameData.ServerActionValid:
                dataOk = True
                print("Action valid!")

                agent.update_trash(data.card)
                agent.hint_gained()

                if data.lastPlayer == agent_name:
                    agent.discover_card(data.card, data.cardHandIndex, 'discard')
                    print("Current player: " + data.player)
                    # possibly notify the condition variable
                    check_agent_turn(data.player)
                else:
                    agent.track_played_card(data.lastPlayer, data.cardHandIndex)
                    if data.handLength == 5:
                        show_action()

            # 6 received when one player plays a card correctly
            if type(data) is GameData.ServerPlayerMoveOk:
                dataOk = True
                print("Nice move!")

                agent.update_board(data.card)

                if data.lastPlayer == agent_name:
                    agent.discover_card(data.card, data.cardHandIndex, 'play')
                    print("Current player: " + data.player)
                    # possibly notify the condition variable
                    check_agent_turn(data.player)
                else:
                    agent.track_played_card(data.lastPlayer, data.cardHandIndex)
                    if data.handLength == 5:
                        show_action()

            # 7 received when one player makes a mistake
            if type(data) is GameData.ServerPlayerThunderStrike:
                dataOk = True
                print("OH NO! The Gods are unhappy with you!")

                agent.update_trash(data.card)
                agent.mistake_made()

                if data.lastPlayer == agent_name:
                    agent.discover_card(data.card, data.cardHandIndex, 'mistake')
                    print("Current player: " + data.player)
                    # possibly notify the condition variable
                    check_agent_turn(data.player)
                else:
                    agent.track_played_card(data.lastPlayer, data.cardHandIndex)
                    if data.handLength == 5:
                        show_action()

            # 8 received when one player hints another player
            # if type(data) is GameData.ServerHintData:
            #     dataOk = True
            #     print("Hint type: " + data.type)
            #     print("Player " + data.destination + " cards with value " + str(data.value) + " are:")
            #     for i in data.positions:
            #         print("\t" + str(i))
            #     agent.update_knowledge_on_hint_received(data)
            #     if data.destination == agent.name:
            #         hint_received = True
            #         show_action()
            #
            #     #########################
            #     # TODO
            #     # * decrementare valore carte nella mano dell'agent quando una è FD tranne la carta FD:
            #     #        -> agent.py, line 135
            #     # * fare in modo che una carta FD faccia scattare l'aggiornamento più di una volta (attributi di classe fully_determined):
            #     #       -> function get_new_fully_determined_cards()
            #     #       -> function reset_recent_fully_determined_cards()
            #     #########################
            #
            #     # check if after THIS hint to agent the following update generated fully determined cards in agent's hand
            #     if data.destination == agent.name:
            #         # retrieve recent fully determined cards
            #         fd_cards = agent.knowledge.player_mental_state(agent.name).get_new_fully_determined_cards()
            #         # fd_cards is a list of card indexes in agent's hand which have been detected Fully Determined recently
            #         print('fully determined cards generated with last hint in agent"s hand:\n')
            #         print(fd_cards)
            #
            #         if (len(fd_cards) != 0):
            #             # if agent has recent Fully Determined cards...
            #             for card_index in fd_cards:
            #                 # ...for each one get the rank and the color of it
            #                 print("Risultato get_card_from_index:\n")
            #                 print(agent.knowledge.player_mental_state(agent.name).get_card_from_index(card_index))
            #                 rank, color = np.nonzero(
            #                     agent.knowledge.player_mental_state(agent.name).get_card_from_index(
            #                         card_index).get_table())
            #                 print("Rank and Color of the FD card: ", rank, color)
            #                 # update mental state of each player (including agent)
            #                 for player in agent.players:
            #                     agent.knowledge.player_mental_state(player).update_whole_hand(rank, color,
            #                                                                                   fully_determined=True)
            #             # now recent fully determined cards are not recent anymore...
            #             agent.knowledge.player_mental_state(agent.name).reset_recent_fully_determined_cards()
            #
            #     show_action()

            # 8 received when one player hints another player
            if type(data) is GameData.ServerHintData:
                dataOk = True
                if DEBUG:
                    print("Hint type: " + data.type)
                    print("Player " + data.destination + " cards with value " + str(data.value) + " are:")
                    for i in data.positions:
                        print("\t" + str(i))

                agent.hint_consumed()
                if data.source != agent.name:
                    agent.update_knowledge_on_hint(data.type, data.value, data.positions, data.destination)

                check_agent_turn(data.player)

            # 9 received when the agent performs an action against the game rules (?)
            if type(data) is GameData.ServerInvalidDataReceived:
                dataOk = True
                if DEBUG:
                    print(data.data)
                with cv:
                    cv.notify()
                # something went wrong, it shouldn't happen

            # 10 received when the game is over for some reason
            if type(data) is GameData.ServerGameOver:
                dataOk = True
                print(data.message)
                print(data.score)
                print(data.scoreMessage)
                stdout.flush()
                run = False
                with cv:
                    cv.notify()
                # TODO: stop client thread (destroy CV ??)

            if not dataOk:
                print("Unknown or unimplemented data type: " + str(type(data)))
            # print("[" + agent_name + " - " + status + "]: ", end="")
            stdout.flush()


if __name__ == '__main__':
    main()
