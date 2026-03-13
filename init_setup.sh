# Create environment
python3 -m venv venv
source venv/bin/activate

# Setup comfy cli with nvidia gpu
pip install comfy-cli
comfy install --nvidia