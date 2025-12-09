import json
with open('configs/settings.json','r') as f:
    settings = json.load(f)
# Expose as SETTINGS
SETTINGS = settings
