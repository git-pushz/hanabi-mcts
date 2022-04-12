#!/usr/bin/env python3
from sys import argv, stdout
from threading import Thread, Condition
import GameData
import socket
from constants import *
from agent import Agent, DEBUG, VERBOSE
import traceback


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
    statuses = ["Lobby", "Game", "GameHint"]
    status = statuses[0]

    def show_action():
        """
        Called to send a `show` request to the server
        """
        if status == statuses[1]:
            s.send(GameData.ClientGetGameStateRequest(agent_name).serialize())

    def agent_move_thread():
        """
        This is the code of the thread that manages the agent. It waits on a condition variable for its turn, calls the
        Agent.make_move method and sends its result to the server.
        """
        with cv:
            while run:
                if DEBUG or VERBOSE:
                    print("waiting on cv")
                cv.wait()  # wait for our turn
                if not run:
                    break
                print(agent.known_status())
                if DEBUG or VERBOSE:
                    print("cv notified!")
                try:
                    move = agent.make_move()
                    if move is not None:
                        if True:
                            print(
                                f"At turn {agent.turn} I chose the move: {move.action}:"
                            )
                            if hasattr(move, "handCardOrdered"):
                                print(f"\tCard: {move.handCardOrdered}")
                            if hasattr(move, "type") and hasattr(move, "value"):
                                print(
                                    f"\tHint: {move.type} {move.value} to {move.destination}"
                                )
                    elif move is None:
                        print("MOVE IS NONE")
                    s.send(move.serialize())
                except Exception:
                    print(traceback.format_exc())
                stdout.flush()

    def check_agent_turn(current_player: str):
        """
        Utility function: checks if current_player is the agent and possibly perform a notify on the condition variable.

        Args:
            current_player: the player of this turn, according to the GameData.ServerToClientData object received.
        """
        if current_player == agent_name:
            with cv:
                cv.notify()

    def check_turn_and_new_cards(agent_obj: Agent, new_card_drawn: bool, last_player: str, current_player: str) -> None:
        """
        Performs the right action based on the last action performed by some player.

        Args:
            agent_obj: the object of class Agent
            new_card_drawn: it's False if the deck was empty and who played couldn't draw a new card
            last_player: the name of the last player
            current_player: the name of the player whose turn is
        """
        if last_player == agent_obj.name:
            if new_card_drawn:
                agent_obj.draw_card()
            print("Current player: " + current_player)
        else:
            if new_card_drawn:
                # trigger a check for the new drawn card and the next player
                show_action()
            else:
                check_agent_turn(current_player)

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
                    print(
                        "Ready: "
                        + str(data.acceptedStartRequests)
                        + "/"
                        + str(data.connectedPlayers)
                        + " players"
                    )

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
                else:
                    agent.track_drawn_card(data.players)
                agent.assert_aligned_with_server(
                    data.usedNoteTokens,
                    data.usedStormTokens,
                    data.tableCards,
                    data.discardPile,
                    data.players,
                )
                check_agent_turn(data.currentPlayer)

            # 4 received when someone performs an invalid action
            if type(data) is GameData.ServerActionInvalid:
                dataOk = True
                # something is wrong, shouldn't be here
                print("Invalid action performed. Reason:")
                print(data.message)
                stdout.flush()
                run = False
                # decrement turn because this notify will make the agent take another decision (in the current turn)
                # in the make_move, which by default increments the turns count
                agent.turn -= len(agent.players)
                with cv:
                    cv.notify()

            # 5 received when one player discards a card
            if type(data) is GameData.ServerActionValid:
                dataOk = True
                print("Action valid!")

                if data.lastPlayer == agent_name:
                    agent.discover_own_card(data.card, data.cardHandIndex)

                agent.track_discarded_card(data.lastPlayer, data.cardHandIndex)

                check_turn_and_new_cards(
                    agent,
                    data.handLength == agent.hand_size,
                    data.lastPlayer,
                    data.player,
                )

            # 6 received when one player plays a card correctly
            if type(data) is GameData.ServerPlayerMoveOk:
                dataOk = True
                print("Nice move!")

                if data.lastPlayer == agent_name:
                    agent.discover_own_card(data.card, data.cardHandIndex)

                agent.track_played_card(
                    data.lastPlayer, data.cardHandIndex, correctly=True
                )

                check_turn_and_new_cards(
                    agent,
                    data.handLength == agent.hand_size,
                    data.lastPlayer,
                    data.player,
                )

            # 7 received when one player makes a mistake
            if type(data) is GameData.ServerPlayerThunderStrike:
                dataOk = True
                print("OH NO! The Gods are unhappy with you!")

                if data.lastPlayer == agent_name:
                    agent.discover_own_card(data.card, data.cardHandIndex)

                agent.track_played_card(
                    data.lastPlayer, data.cardHandIndex, correctly=False
                )

                check_turn_and_new_cards(
                    agent,
                    data.handLength == agent.hand_size,
                    data.lastPlayer,
                    data.player,
                )

            # 8 received when one player hints another player
            if type(data) is GameData.ServerHintData:
                dataOk = True
                if DEBUG:
                    print("Hint type: " + data.type)
                    print(
                        "Player "
                        + data.destination
                        + " cards with value "
                        + str(data.value)
                        + " are:"
                    )
                    for i in data.positions:
                        print("\t" + str(i))

                agent.track_hint(
                    data.destination, data.positions, data.type, data.value
                )

                check_agent_turn(data.player)

            # 9 received when the agent performs an action against the game rules (?)
            if type(data) is GameData.ServerInvalidDataReceived:
                dataOk = True
                if DEBUG:
                    print(data.data)
                # decrement turn because this notify will make the agent take another decision (in the current turn)
                # in the make_move, which by default increments the turns count
                agent.turn -= 1
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

            if not dataOk:
                print("Unknown or unimplemented data type: " + str(type(data)))
            # print("[" + agent_name + " - " + status + "]: ", end="")
            stdout.flush()


if __name__ == "__main__":
    main()
