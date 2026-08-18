# LCLS-SC Quench Data Analysis

This project contains Python tools for analyzing, visualizing, and labeling SRF cavity quench data from the LCLS-SC. It is built using Python and various Python libraries. It includes:

- A **plotting toolkit** for generating quench analysis plots from `.h5` data files.
- A **Streamlit labeling interface** for viewing and classifying quench waveforms.

---

## Prerequisites

- **Python 3.12 or newer**
- **Git** installed (to clone the repository)
- **Conda** install is recommended
- Your own `.h5` quench data file(s) *(not included in this repo)*

To check if you have Python and Git:

```bash
python --version
git --version
```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/slaclab/slacq.git
cd slacq
```

### 2. Directory Structure

```
slacq/
├── classification/
├── config/
├── interface/
├── plotter/
├── tests/
├── utils/
├── .gitignore
├── AGENTS.md
├── README.md
├── generate_plots.py
└── pyproject.toml
```

- Create a `data/` directory (not tracked in the repo) and place your `.h5` files in it.
- All plotting scripts are located in the `plotter` package.
- All `CSV` files containing multipacting dates are located in the `config` directory.
- Classification function lives in `classification/`.
- Shared helper code lives in `utils/`.
- Tests live in `tests/`.
- The Streamlit labeling app lives in `interface/`.
- Output plots are saved to an `images/` directory, created automatically when you run the plotting script.

### 3. Create and Activate a Conda Environment

```bash
conda create -n YOUR_ENV_NAME python=3.12
conda activate YOUR_ENV_NAME
```

### 4. Install Required Python Packages

```bash
conda install numpy pandas h5py matplotlib scipy streamlit plotly
```

---

## Usage

### Quench Plot Generation

1. Run the script:
   ```bash
   python generate_plots.py
   ```
2. Find the output — generated plot files will be saved in the `images/` folder.

### Classification

Run this script from the root project directory using:

```bash
python -m classification.evaluate
```

### Quench Labeling Interface

> **Everything below (through Troubleshooting) applies to the Streamlit labeling interface.**

The labeling interface is a Streamlit web app for viewing and labeling cavity quench waveforms stored in HDF5 files.

1. **Get a data file.** The app needs an HDF5 (`.h5`) quench data file, which is not included in the repository. Example path:
   ```bash
   /Users/yourname/data/quench_data_L1.h5
   ```

2. **Run the app**
   ```bash
   streamlit run interface/app.py
   ```
   > Use `streamlit run`, not `python interface/app.py`.

   The app opens in your browser automatically. If it doesn't, copy the **URL** from the terminal into your browser.

3. **Use the app:**
   - Enter the full path to your `.h5` file in the text box at the top.
   - Filter events by **cryomodule, cavity, year**, and **label**.
   - Select an event to view its waveform and classification suggestion.
   - Label the event using the buttons at the bottom.

   ***Tip:*** When labeling, set the filter to "unlabeled." This way, once you label an event, the app automatically moves you to the next unlabeled one.

4. **Stop the app** — in the terminal, press:
   ```
   Ctrl + C
   ```

---

#### Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Activate your conda environment with `conda activate YOUR_ENV_NAME` and try again. If it's still not working, install Streamlit: `conda install streamlit` |
| Nothing happens when you run it | You used `python` instead of `streamlit run` |
| File not found in the app | The HDF5 path you entered is incorrect |
| `ModuleNotFoundError` | Make sure you are in the `root` directory |
| App won't open in browser | Manually paste the URL from the terminal into your browser |