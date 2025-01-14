#!/bin/bash

# # Remote server details
# REMOTE_USER="msekatchev"
# REMOTE_HOST="consus.phas.ubc.ca"
# REMOTE_PORT=7743
# REMOTE_DIR="/data4/msekatchev/FIREbackup08012025"

# # Local files and directory to transfer
# LOCAL_DIR="data/FIRE/cubes/"
# FILES=(
#     "data/FIRE/grid-ids--dark_mat-cKDTree.npy"
#     "data/FIRE/grid-ids--neut_gas-cKDTree.npy"
#     "data/FIRE/grid-ids--ioni_gas-cKDTree.npy"
#     "data/FIRE/grid-coords.npy"
#     "data/FIRE/part.pkl"
# )

# # Create the remote directory (if it doesn't exist)
# echo "Creating remote directory: $REMOTE_DIR"
# ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR"

# # Transfer directory
# echo "Uploading directory $LOCAL_DIR..."
# scp -P $REMOTE_PORT -r $LOCAL_DIR $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR
# echo "Directory $LOCAL_DIR uploaded successfully."

# # Transfer individual files
# for file in "${FILES[@]}"; do
#     echo "Uploading file $file..."
#     scp -P $REMOTE_PORT "$file" $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR
#     if [[ $? -eq 0 ]]; then
#         echo "File $file uploaded successfully."
#     else
#         echo "Error uploading $file."
#     fi
# done

# echo "Backup completed successfully."

#### New version using RSYNC

# Define variables
LOCAL_DATA_DIR="data"                    # Path to the local 'data' folder
REMOTE_USER="msekatchev"                 # Your username on Consus
REMOTE_HOST="consus.phas.ubc.ca"         # Consus server address
REMOTE_PORT="7743"                       # Custom SSH port
REMOTE_DIR="/data4/msekatchev/data"      # Remote destination directory

# rsync command with custom SSH port
rsync -avzP --delete -e "ssh -p $REMOTE_PORT" "$LOCAL_DATA_DIR/" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/"

# Print success message
if [ $? -eq 0 ]; then
    echo "Data folder successfully synced to $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
else
    echo "Sync failed. Check the error messages above."
    exit 1
fi

