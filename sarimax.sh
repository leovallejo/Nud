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
    curl -s -H "Authorization: token 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" \
     https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/app.py \
     -o /root/allora-huggingface-walkthrough/app.py && \
    curl -s -H "Authorization: token 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" \
     "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/requirements.txt" \
     -o /root/allora-huggingface-walkthrough/requirements.txt && \
     curl -s -H "Authorization: token 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" \
     "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/requirements.txt" \
     -o /root/allora-huggingface-walkthrough/model.py && \
     curl -s -H "Authorization: token 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" \
     "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/requirements.txt" \
     -o /root/allora-huggingface-walkthrough/api_client.py
    
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
