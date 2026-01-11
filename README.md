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

## Setup

### 1. Create/Activate Conda Environment

When first accesing Habrok, the conda module needs to be activated.
```bash
module load Anaconda3
```

Now to create/activate an environment.
```bash
conda create -n biomistral_app python=3.11
```

```bash
conda activate biomistral_app
```

The module load and conda activate commands will have to be reused every time when restarting.

### 2. Install Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

This will install the libraries in your conda environment. One-time use.

### 3. Request GPU Resources

```bash
srun --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=01:00:00 --pty bash
```

## Running the Application

The app consists of two components: the **LLM Server** and the **Flask Web App**.
Make sure you open new terminals for each server.

### Terminal 1: Start the LLM Server

```bash
conda activate biomistral_demo
python -u LLM/llm_server.py
```

### Terminal 2: Start the Flask Web App

```bash
conda activate biomistral_demo
python App/webapp/app.py
```

## Accessing the Application
### Terminal 3: SSH Port Forwarding

From your local machine, create an SSH tunnel:
```bash
ssh -L 5000:localhost:5000 s....@login-node-hostname
```
s... refers to the student number used to login into habrok
login-node-hostname should be replaced with the active habrok node you were allocated.
It is found after your s/p number in the terminal after the "@".
e.g:[s5068290@a100gpu2 Bachelor-s-Project]$ ssh -L 5000:localhost:5000 s5068290@a100gpu2

Then access:
```
http://localhost:5000
```