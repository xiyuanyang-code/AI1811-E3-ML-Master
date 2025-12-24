echo "Killing ML-Master"
ps -x | grep mcts | grep -v grep | awk '{print $1}' | xargs kill -9