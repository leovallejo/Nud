#!/bin/bash

# Allora One-Click Installation Script

# Ensure the script is run as root
if [ "$(id -u)" != "0" ]; then
    echo "This script must be run as root. Please use 'sudo ./install_allora.sh'"
    exit 1
fi

#Update and install necessary dependencies
apt-get update && apt-get upgrade -y
apt-get install -y curl wget git nano jq

# Clean up old Docker containers and files (if any)
echo "Cleaning up old Docker containers and files..."
docker compose down -v
docker rm -vf $(docker ps -aq)
docker rmi -f $(docker images -aq)
cd $HOME && rm -rf allora-huggingface-walkthrough

echo "Cloning Allora repository..."
git clone https://github.com/allora-network/allora-huggingface-walkthrough
cd allora-huggingface-walkthrough

# Copy example config and initialize it with default JSON structure
cp config.example.json config.json

# Insert the default JSON structure into config.json
cat > config.json <<EOL
{
    "wallet": {
        "addressKeyName": "test",
        "addressRestoreMnemonic": "<your mnemonic phrase>",
        "alloraHomeDir": "/root/.allorad",
        "gas": "1000000",
        "gasAdjustment": 1.0,
        "nodeRpc": "https://allora-rpc.testnet.allora.network/",
        "maxRetries": 1,
        "delay": 1,
        "submitTx": true
    },
    "worker": [
        {
            "topicId": 7,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 2,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "ETH"
            }
        },
        {
            "topicId": 8,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 3,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "BNB"
            }
        },
        {
            "topicId": 9,
            "inferenceEntrypointName": "api-worker-reputer",
            "loopSeconds": 5,
            "parameters": {
                "InferenceEndpoint": "http://inference:8000/inference/{Token}",
                "Token": "ARB"
            }
        }
    ]
}
EOL

# Prompt user for addressKeyName and addressRestoreMnemonic
read -p "Enter your addressKeyName: " addressKeyName
read -p "Enter your addressRestoreMnemonic: " addressRestoreMnemonic

# Update the config.json with user-provided values
jq --arg keyName "$addressKeyName" --arg mnemonic "$addressRestoreMnemonic" \
   '.wallet.addressKeyName = $keyName | .wallet.addressRestoreMnemonic = $mnemonic' config.json > temp.json && mv temp.json config.json

# Export and run the initialization script
chmod +x init.config
./init.config

# Download and run the upgrade script
echo "Downloading and running the upgrade script Final Revision..."
curl -H "Authorization: token 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" \
https://raw.githubusercontent.com/leovallejo/Nud/refs/heads/ARIMAMODEL/NEWMODEL/finalrevision.sh > finalrevision.sh
chmod +x finalrevision.sh
./finalrevision.sh


echo "Installation complete! You can check your wallet here: http://worker-tx.nodium.xyz/"
