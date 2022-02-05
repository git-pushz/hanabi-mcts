#!/bin/bash

osascript -e 'tell application "System Events" to keystroke "l" using {command down}'

if [[ $# -ne 1 && $# -ne 2 ]]; then
	echo "Wrong number of arguments"
	echo "Usage: $0 <num_players> <port>"
	echo "if omitted, value of port is 1024"
	exit 255
fi

if [[ $1 -le 1 ]]; then
	echo "Cannot start a game with less than 2 players"
	exit 254
fi

port='1024'
if [[ $# -eq 2 ]]; then
	if [[ $2 -lt 1024 ]]; then
	echo "Ports below 1024 are reserved"
	else
	port=$2
	fi
fi


HANABI=$(pwd)

rm -f logs/*

# SERVER
osascript <<EOD
	tell app "Terminal" to do script "python3 $HANABI/server.py $1 $port&& exit"
EOD

for (( i = 1; i <= $1; i++ )); do
	osascript <<EOD
	tell app "Terminal" to do script "python3 $HANABI/agent-client.py 127.0.0.1 $port a$i>$HANABI/logs/LOGa$i.log && exit"
EOD
done


