# LTP
Repository for the Language Technology Project study on fallacy detection. 

*Contributors:* \
Lucas Velvis \
Minne Schalekamp \
Ivo Bruinier \
Xuechun Wu


## Running the code
First, install packages: \
`pip install -r requirements.txt`

For running the code there are several options:
- Get an example prompt of a single prompting technique:
`python3 main.py [technique]`.
    - `[technique]` options: `zero_shot`, `few_shot`, `auto_cot`, `gen_knowledge`.
- Just run the experiment: `python3 main.py experiment`.
- Run the experiment including evaluation (with F1-score): `python3 main.py complete`.

Simply running `python3 main.py` will default to the `complete` run.

## General Implementation
The main models around which this project is built are the Large Language Models given in `models` and the prompting techniques in the folder `prompting techniques`. Each containing a template superclass in `model.py` and `prompt.py` respectively. The `standard_format` of the prompt superclass is based on the prompts provided in the MAFALDA paper [[1]](#1).

`data.py` contains a simple `Data` class which consists of a text-string and a label (with end and start indices) provided by the `Label` class.

In `experiment.py` the experiment is run by applying each supplied prompting technique to each supplied model for each question in the Gold Standard Dataset (GSD), storing the results in the same format as the GSD for easy evaluation. 

In `evaluation.py` the results are exctracted from the data files and evaluated based on the adapted F1-score as provided in the MAFALDA paper [[1]](#1).

`fallacy_extraction.py` provides the code that is used to extract the labelled fallacy from an LLM response. The keywords used in this file have been retrieved from the code of the MAFALDA paper [[1]](#1), at [Github](https://github.com/ChadiHelwe/MAFALDA).

## Dataset Acknowledgements
For evaluation, this study makes use of the Gold Standard Dataset that was created by the authors of the MAFALDA paper [[1]](#1), and was retrieved from [Github](https://github.com/ChadiHelwe/MAFALDA).

For constructing the "Generated Knowledge" prompts, the [knowlede_gpt3.dev.csqa] dataset created by J. Liu et al. [[2]](#2), was used, retrieved from [Github](https://github.com/liujch1998/GKP).

## Model Acknowledgements
- Falcon-7b LLM by E. Almazrouei et al. [[3]](#3)
- Zephyr-7b LLM by L. Tunstall et al. [[4]](#4)

## References
<a id="1">[1]</a> 
C. Helwe, T. Calamai, P.-H. Paris, C. Clavel, and F. Suchanek, ‘MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection and Classification’, in Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2024.

<a id="2">[2]</a> 
J. Liu et al., ‘Generated knowledge prompting for commonsense reasoning’, arXiv preprint arXiv:2110. 08387, 2021.

<a id="3">[3]</a> 
E. Almazrouei et al., “Falcon-40B: an open large language model with state-of-the-art performance,” 2023.

<a id="4">[4]</a> 
L. Tunstall et al., Zephyr: Direct Distillation of LM Alignment. 2023.