
#!/bin/bash

# Add all files except those in the data/FIRE directory
git add --all -- ':!data/FIRE'

# Commit changes with the provided message
git commit -m "$1"

# Push changes to remote repository
git push
