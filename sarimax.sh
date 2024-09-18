#!/bin/bash

BOLD="\033[1m"
UNDERLINE="\033[4m"
LIGHT_BLUE="\033[1;34m"     # Light Blue for primary messages
BRIGHT_GREEN="\033[1;32m"   # Bright Green for success messages
MAGENTA="\033[1;35m"        # Magenta for titles
RESET="\033[0m"             # Reset to default color

echo -e "${LIGHT_BLUE}Upgrade Your Allora Model to SARIMAX(Y/N):${RESET}"
read -p "" installdep
echo

if [[ "$installdep" =~ ^[Yy]$ ]]; then

    echo -e "${LIGHT_BLUE}Clone & Replace old file :${RESET}"
    echo
    rm -rf app.py
    rm -rf requirements.txt
    wget -q https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/app.py?token=GHSAT0AAAAAACW4AWHZNHQXPQSJTQAIWDZUZXKEM6Q -O /root/allora-huggingface-walkthrough/app.py
    wget -q https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/requirements.txt?token=GHSAT0AAAAAACW4AWHZOGOEUX5IRVZ2HFPGZXKELEQ -O /root/allora-huggingface-walkthrough/requirements.txt
    wait
	
    echo -e "${LIGHT_BLUE}Rebuild and run a model :${RESET}"

    cd /root/allora-huggingface-walkthrough/	
    echo
    docker compose up --build -d
    echo
	
    echo
    docker compose logs -f 
    echo
	
else
    echo -e "${BRIGHT_GREEN}Operation Canceled :${RESET}"
    
fi

echo
echo -e "${MAGENTA}==============thanks to 0xTnpxSGT | Allora SARIMAX===============${RESET}"
