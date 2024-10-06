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
   # rm -rf .env
    rm -rf app.py
    rm -rf requirements.txt
   #NEW MODEL
   curl -s -H "Authorization: token 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" \
     https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/NEWMODEL/app.py \
     -o /root/allora-huggingface-walkthrough/app.py && \
    curl -s -H "Authorization: token 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" \
     "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/NEWMODEL/requirements.txt" \
     -o /root/allora-huggingface-walkthrough/requirements.txt && \
     #curl -s -H "Authorization: token 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" \
     #"https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/NEWMODEL/.env" \
     #-o /root/allora-huggingface-walkthrough/.env
     
     curl -s -H "Authorization: token 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" \
     "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/NEWMODEL/Dockerfile" \
     -o /root/allora-huggingface-walkthrough/Dockerfile && \
     
     curl -s -H "Authorization: token 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" \
     "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/NEWMODEL/Dockerfile" \
     -o /root/allora-huggingface-walkthrough/Dockerfile
     
    #FOR CNN MODEL
   # curl -s -H "Authorization: token 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" \
    # https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/app.py \
    # -o /root/allora-huggingface-walkthrough/app.py && \
   # curl -s -H "Authorization: token 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" \
   #  "https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/requirements.txt" \
  #   -o /root/allora-huggingface-walkthrough/requirements.txt 

     
    
    wait
	
    echo -e "${LIGHT_BLUE}Rebuild and run a model :${RESET}"

    cd /root/allora-huggingface-walkthrough/	
    echo
    docker compose up --build -d
    echo
	
    echo
    docker compose logs -f inference-hf
    echo
	
else
    echo -e "${BRIGHT_GREEN}Operation Canceled :${RESET}"
    
fi

echo
echo -e "${MAGENTA}==============thanks to 0xTnpxSGT | Allora SARIMAX===============${RESET}"
