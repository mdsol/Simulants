# Simulants 

In order to address the privacy concerns of patient data and to be able to disclose clinical trial data to
other organizations, we have built a system that synthesizes patient data and cross-validates the synthetic data
against the real data by running standard statistical techniques and machine learning algorithms.
The code consists of a set of libraries used for loading sample data from the UCI reposirtory, preprocessing it
and using it to synthesize a new set of patients.

A sample dataset is downloaded from the UCI Machine Learning Repository at:
https://archive.ics.uci.edu/ml/datasets/Heart+Disease


## Prerequisites
use python 3.8 or later

All the required packages are specified in requirements.txt.

pip install -r requirements.txt



## Usage
1. Modify uci_config.py    or use it as it for using the sample dataset from uci heart disease

2. python uci_demo.py

3. the outputs ncluding the synthesized data and the results from cross-validation will be in output_uci/


## Contributing
See [CONTRIBUTING](CONTRIBUTING.md).

## Contributors
Jacob Aptekar
Mandis Beigi
Pierre-Louis Bourlon
Jason Mezey
Afrah Shafquat


## Contact
See the [factbook](factbook.yaml).
## Contact
Mandis Beigi at AcornAI (Medidata Solutions Inc., a Dassault Systemes Company)

mandis.beigi@3ds.com

