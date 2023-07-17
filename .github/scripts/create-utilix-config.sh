#!/bin/bash

# Create config with write permissions!
cat > $HOME/.xenon_config <<EOF
[RunDB]
rundb_api_url = $RUNDB_API_URL
rundb_api_user = $RUNDB_API_USER
rundb_api_password = $RUNDB_API_PASSWORD
xent_url = $XENT_URL
xent_user = $XENT_USER
xent_password = $XENT_PASSWORD
xent_database = $XENT_DATABASE
EOF