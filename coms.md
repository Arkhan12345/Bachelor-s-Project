module load Anaconda3
conda activate biomistral_demo
srun --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=04:00:00 --pty bash
tmux new -s biomistral
cd ~/Bproj/Bachelor-s-Project/Bachelor-s-Project/LLM
python llm_server.py
cd ~/Bproj/Bachelor-s-Project/Bachelor-s-Project/App/webapp
python app.py

ssh -N -L 15000:v100v2gpu16:5000 -L 18000:v100v2gpu16:8000 s5068290@interactive1.hb.hpc.rug.nl
