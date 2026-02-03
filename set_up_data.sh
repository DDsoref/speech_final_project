#!/bin/bash

# 1. Create Data Directory
echo "Creating data folder..."
mkdir -p data
cd data

# --- PART 1: NOISEX-92 SETUP ---
echo "Downloading NOISEX-92..."
git clone https://github.com/speechdnn/Noises.git

echo "Sorting noises into Pre-train and Incremental..."
mkdir -p pretrain_noises
mkdir -p incremental_noises

# Move the 5 "Incremental" tasks (New Tasks)
# Includes our substitutes: f16->Alarm, volvo->Cough, m109->Extra
cp Noises/NoiseX-92/destroyerops.wav incremental_noises/
cp Noises/NoiseX-92/machinegun.wav incremental_noises/
cp Noises/NoiseX-92/f16.wav incremental_noises/
cp Noises/NoiseX-92/volvo.wav incremental_noises/
cp Noises/NoiseX-92/m109.wav incremental_noises/

# Move the 10 "Base" tasks (Pre-train)
cp Noises/NoiseX-92/babble.wav pretrain_noises/
cp Noises/NoiseX-92/buccaneer1.wav pretrain_noises/
cp Noises/NoiseX-92/buccaneer2.wav pretrain_noises/
cp Noises/NoiseX-92/destroyerengine.wav pretrain_noises/
cp Noises/NoiseX-92/factory1.wav pretrain_noises/
cp Noises/NoiseX-92/factory2.wav pretrain_noises/
cp Noises/NoiseX-92/hfchannel.wav pretrain_noises/
cp Noises/NoiseX-92/leopard.wav pretrain_noises/
cp Noises/NoiseX-92/pink.wav pretrain_noises/
cp Noises/NoiseX-92/white.wav pretrain_noises/

# --- PART 2: LIBRISPEECH SETUP ---
echo "Downloading LibriSpeech (This may take time)..."

# Dev-Clean (Validation)
curl -L -O https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzvf dev-clean.tar.gz
rm dev-clean.tar.gz

# Test-Clean (Final Exam)
curl -L -O https://www.openslr.org/resources/12/test-clean.tar.gz
tar -xzvf test-clean.tar.gz
rm test-clean.tar.gz

# Train-Clean-100 (Training Data)
curl -L -O https://www.openslr.org/resources/12/train-clean-100.tar.gz
tar -xzvf train-clean-100.tar.gz
rm train-clean-100.tar.gz

echo "Done! Your data folder is ready."