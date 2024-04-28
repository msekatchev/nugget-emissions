
#!/bin/bash

# Add all files except those in the data/FIRE directory
git add --all -- ':!data/FIRE' ':!data/skymap_integration_cube_indexes.npy'

# Commit changes with the provided message
git commit -m "$1"

# Push changes to remote repository
git push
