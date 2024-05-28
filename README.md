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

## Acknowledgements
For evaluation, this study makes use of the Gold Standard Dataset that was created by the authors of the MAFALDA paper [[2]](#2), and was retrieved from [Github](https://github.com/ChadiHelwe/MAFALDA). The adapted F1-score evaluation was also based on this paper.

For constructing the "Generated Knowledge" prompts, the [knowlede_gpt3.dev.csqa] dataset created by J. Liu et al. [[1]](#1), was used, retrieved from [Github](https://github.com/liujch1998/GKP).

## References
<a id="1">[1]</a> 
J. Liu et al., ‘Generated knowledge prompting for commonsense reasoning’, arXiv preprint arXiv:2110. 08387, 2021.

<a id="1">[2]</a> 
C. Helwe, T. Calamai, P.-H. Paris, C. Clavel, and F. Suchanek, ‘MAFALDA: A Benchmark and Comprehensive Study of Fallacy Detection and Classification’, in Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2024.
