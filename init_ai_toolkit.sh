git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
python3 -m venv aivenv
source aivenv/bin/activate

# install torch first
pip3 install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt