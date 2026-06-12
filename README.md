# Bachelor's Project

A biomedical research assistant application that integrates Independent Component Analysis (ICA) with a BioMistral LLM for gene pathway analysis and literature summarization.

## Project Structure

```
App/
  webapp/
    app.py              # Flask web application (port 5000)
    static/             # CSS, JS files
    templates/          # HTML templates
  pipeline.py           # ICA analysis pipeline
  plot_annotations.py   # Visualization functions 

LLM/
  llm_server.py         # LLM API server (port 8000)
  BioMistral_app.py     # Alternative Flask chat interface
  Biomistral_demo.py    # BioMistral model functions

Archive/                # Reference data files
```

---

## Correct run instructions

### How to connect to a SSH host

On VSCode, install the Remote SSH Extension.
To connect to the cluster press Ctrl + Shift + P and type "Remote-SSH: Connect to Host".
Select "Add New SSH Host" and enter "ssh <USERNAME>@interactive1.hb.hpc.rug.nl".
VS Code will ask where to save the SSH configuration.
Choose the default file.

Now connect to the cluster:

"Remote-SSH: Connect to Host"

Select your required host and enter your password.
After connecting, VS Code will open a new window connected to the cluster.

In the remote VS Code window clone the github repository.

**Important idea:** run BOTH servers (LLM on 8000 and Webapp on 5000) on the **same GPU compute node** (the one you get from `srun`).  
Only do SSH port-forwarding from your **laptop → cluster**.

### Step 0 — (One-time) create conda env
Run this once (on any node where you have conda available; typically `interactive1` is easiest):
```bash
module load Anaconda3
conda create -n biomistral_demo python=3.11 -y
conda activate biomistral_demo
pip install -r ~/Bproj/Bachelor-s-Project/Bachelor-s-Project/requirements.txt
```

After this, you will normally only need `module load` + `conda activate`.

---

### Step 1 — Request GPU resources
```bash
srun --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=01:00:00 --pty bash
```

Change values accordingly.
You should now be on a compute node (example):
```
[s5068290@a100gpu6 ...]$
```

---

### Step 2 — Activate environment (on the compute node)
```bash
module load Anaconda3
conda activate biomistral_demo
```

---

### Step 3 — Start tmux (on the compute node)
```bash
tmux new -s biomistral
```

Useful tmux keys:
- Split panes: `Ctrl+b` then `%` (vertical) or `"` (horizontal)
- Switch panes: `Ctrl+b` then arrow keys
- Detach: `Ctrl+b` then `d`
- Re-attach later: `tmux attach -t biomistral`

---

### Step 4 — Start the LLM server (tmux pane 1, compute node)
```bash
cd ~/Bproj/Bachelor-s-Project/Bachelor-s-Project/LLM
python llm_server.py
```

You should see something like:
```
Running on http://127.0.0.1:8000
```

(Optional check in another pane)
```bash
ss -tulpn | grep ':8000'
```

---

### Step 5 — Start the Flask webapp (tmux pane 2, compute node)
```bash
cd ~/Bproj/Bachelor-s-Project/Bachelor-s-Project/App/webapp
python app.py
```

You should see something like:
```
Running on http://127.0.0.1:5000
```

(Optional check in another pane)
```bash
ss -tulpn | grep ':5000'
```

---

### Step 6 — Local computer port-forward (run ONLY on your local computer)
Open a new terminal on your **computer** and run:

```bash
ssh -N -L 5000:<COMPUTE_NODE>:5000 -L 8000:<COMPUTE_NODE>:8000 s5068290@interactive1.hb.hpc.rug.nl
```

Replace `<COMPUTE_NODE>` with the node you got in Step 1 (e.g. `a100gpu6`).

**Do not run multiple `ssh -L ...` commands.** You only need **one** tunnel from your laptop.

---

### Step 7 — Open the app in your browser (on your laptop)
Visit:
- http://127.0.0.1:5000
(or `http://localhost:5000`)

---
