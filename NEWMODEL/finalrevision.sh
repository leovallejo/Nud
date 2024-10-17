#!/bin/bash

BOLD="\033[1m"
UNDERLINE="\033[4m"
LIGHT_BLUE="\033[1;34m"     # Light Blue for primary messages
BRIGHT_GREEN="\033[1;32m"   # Bright Green for success messages
MAGENTA="\033[1;35m"        # Magenta for titles
RESET="\033[0m"             # Reset to default color

echo -e "${LIGHT_BLUE}Upgrade Your Allora Model to Final Revision(Y/N):${RESET}"
read -p "" installdep
echo

if [[ "$installdep" =~ ^[Yy]$ ]]; then

    echo -e "${LIGHT_BLUE}Clone & Replace old file :${RESET}"
    echo
    rm -rf requirements.txt
    rm -rf app.py
    rm -rf docker-compose.yaml
    rm -rf Dockerfile
    curl -s -H "Authorization: token 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" \
     https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/NEWMODEL/app.py \
     -o /root/allora-huggingface-walkthrough/app.py && \
    curl -s -H "Authorization: token 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" \
     "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/requirements.txt" \
     -o /root/allora-huggingface-walkthrough/requirements.txt && \
    curl -s -H "Authorization: token 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" \
     "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/NEWMODEL/docker-compose.yaml" \
     -o /root/allora-huggingface-walkthrough/docker-compose.yaml && \
    curl -s -H "Authorization: token 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" \
     "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/NEWMODEL/Dockerfile" \
     -o /root/allora-huggingface-walkthrough/Dockerfile
    
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
