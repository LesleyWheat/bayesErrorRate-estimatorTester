# Evaluating Bayes Error Rate Estimators
The purpose of this program is to evaluate Bayes Error Rate (BER) estimators on different synthetic datasets with Monte Carlo simulation.

## How-To: Run the simulations

### (1) Installation
- Install python (3.10 or above)
- Create a virtual enviroment
- Activate the virtual environment
- Configure the settings in config.py
- Run the install script 'python install.py'

To create graphs:
- Install Latex

### (2) Run tests

Run general tests: 'pytest'
To run all tests, including slow and GPU: 'pytest -m ""'

### (3) Run simulations

Run Monte Carlo Simulations: 'python runExp.py'

### (4) Generate graphs and tables

Create tables and graphs: 'python displayPaper.py'.
Note that LaTex is required to make plots and images of tables.

## Attribution
This project redistributes code from the following projects:

- Project: Feebee
  - Author(s): Cedric Renggli, Luka Rimanic, Nora Hollenstein, Ce Zhang
  - Source: https://github.com/DS3Lab/feebee
- Project: LAKDE
  - Author(s): Kenny Falkær Olsen
  - Source: https://github.com/falkaer/lakde
- Project: Tensorflow Hub
  - Author(s): The TensorFlow Hub Authors
  - Source: https://github.com/tensorflow/hub

## Citation
Cite as L. Wheat, M. V. Mohrenschildt, and S. Habibi, “Bayes error rate estimation in difficult situations,” Unpublished, https://doi.org/10.48550/arXiv.2506.03159
