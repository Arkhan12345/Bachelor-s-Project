# Bachelor's Project

A biomedical research assistant application that integrates Independent Component Analysis (ICA) with a BioMistral LLM for gene pathway analysis and literature summarization.

## Project Structure

```
App/
  webapp/
    app.py              # Flask web application
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

### Setup Instructions

### Step 1: Request GPU Resources (terminal 1)

```bash
srun --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=01:00:00 --pty bash
```
Adjust time as needed(hours:minutes:seconds).

#### Step 2: Set up SSH Port Forwarding (terminal 2)

Open a new terminal(from now on use this one) or use the local command prompt to connect with port forwarding:

```bash
ssh -L 5000:v100v2gpu17:5000 -L 8000:v100v2gpu17:8000 s5068290@interactive1.hb.hpc.rug.nl
ssh -L 5000:interactive1:5000 -L 8000:interactive1:8000 s5068290@interactive1.hb.hpc.rug.nl
```

To close the connection:
```bash
exit
```

**Note:** Replace `v100v2gpu17` with your actual compute node name. You'll see it in your terminal prompt (e.g., `[s5068290@v100v2gpu17 ~]$`)
Replace `s5068290` with the student/professor number used to login into habrok.
`interactive1.hb.hpc.rug.nl` represents the login node, replace as needed.

#### Step 3: Create/Activate Conda Environment + Install requirements

```bash
module load Anaconda3
conda create -n biomistral_app python=3.11
conda activate biomistral_app
```

The module load and conda activate(not create) commands will have to be reused every time when restarting.
You will know it is active if you see (biomistral_app) before the path in the terminal.

```bash
pip install -r requirements.txt
```

This will install the libraries in your conda environment. One-time use.


#### Step 4: Navigate to Project Root
e.g.
```bash
cd ~/Bproj/Bachelor-s-Project/Bachelor-s-Project
```

Use `mkdir` command to create a new folder to save this in.
`Bachelor-s-Project` is an example only. Change with the name of your created directory.

#### Step 5: Start LLM Server

```bash
cd LLM
python llm_server.py > llm_server.log 2>&1 &
```

#### Step 6: Start Flask Web App

```bash
cd ../App/webapp
python app.py > webapp.log 2>&1 &
```

#### Step 7: Verify Both Services are Running

```bash
tail webapp.log
```

You should see:
```
* Running on http://...:5000
* Running on http://...:5000
```

#### Step 8: Access the Application

Open a browser on your local machine and go to:
```
http://127.0.0.1:5000/
```
or
```
http://localhost:5000/
```
