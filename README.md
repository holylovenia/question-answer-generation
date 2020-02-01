# Automatic Question-Answer Pairs Generation from Text

## Members

1. [Holy Lovenia](https://github.com/holylovenia)

2. [Agus Gunawan](https://github.com/agusgun)

3. [Felix Limanta](https://github.com/felixlimanta)

## English Paper

We have a technical report that explains our question-answer generation system [here](https://www.researchgate.net/publication/328916588_Automatic_Question-Answer_Pairs_Generation_from_Text).

## Architecture in overview

We propose an approach which is generally based on the framework of an ongoing work by [A. Sarvaiya](https://software.intel.com/en-us/articles/using-natural-language-processing-for-smart-question-generation). Formally, given a passage <img src="https://latex.codecogs.com/gif.latex?P" />, question-answer generation (**QAG**) system retrieves the most important sentence <img src="https://latex.codecogs.com/gif.latex?S" /> from <img src="https://latex.codecogs.com/gif.latex?P" />. Then, **QAG** system produces a set of question-answer pairs <img src="https://latex.codecogs.com/gif.latex?\{(Q_j,&space;A_j)\}" />, where each generated <img src="https://latex.codecogs.com/gif.latex?A_j"/> can be found in <img src="https://latex.codecogs.com/gif.latex?S"/>, and its pair <img src="https://latex.codecogs.com/gif.latex?Q_j"/> is the interrogative version of <img src="https://latex.codecogs.com/gif.latex?S"/> or a clause <img src="https://latex.codecogs.com/gif.latex?C_k"/> from a set of clauses <img src="https://latex.codecogs.com/gif.latex?\{C_k\}"/> in <img src="https://latex.codecogs.com/gif.latex?S"/>, without <img src="https://latex.codecogs.com/gif.latex?A_j"/> in it. As shown in the figure below, here are four main modules  in our **QAG** system.

![architecture-overview](assets/qag-architecture-simple.jpg)

1. **Preprocessing**, which cleans the input passage <img src="https://latex.codecogs.com/gif.latex?P" /> from unnecessary characters and shapes it into the desirable form (list of sentences).

2. **Sentence Selection**, which picks top-<img src="https://latex.codecogs.com/gif.latex?N" /> most important sentences <img src="https://latex.codecogs.com/gif.latex?\{S_1,&space;...,&space;S_N\}" /> given <img src="https://latex.codecogs.com/gif.latex?P" />. The text summarization method used can be chosen between TextRank, multi-word phrase extraction (MWPE), and latent semantic analysis (LSA). The chosen method ranks the sentences in P and selects the top-<img src="https://latex.codecogs.com/gif.latex?N" /> highest ranked sentences as the output.

3. **Gap Selection**, which selects phrases in <img src="https://latex.codecogs.com/gif.latex?S" /> that can be used as answers <img src="https://latex.codecogs.com/gif.latex?\{A_j\}" /> based on constituent tree from syntactic parser and named entity recognition (NER).

4. **Question Formation**, which creates the interrogative version of <img src="https://latex.codecogs.com/gif.latex?S" /> or <img src="https://latex.codecogs.com/gif.latex?C_k&space;\in&space;\{C_k\}" /> in <img src="https://latex.codecogs.com/gif.latex?S" /> without <img src="https://latex.codecogs.com/gif.latex?A_j" /> to make a question <img src="https://latex.codecogs.com/gif.latex?Q_j" /> for each answer in <img src="https://latex.codecogs.com/gif.latex?\{A_j\}" />. The final output of this module is question-answer pairs <img src="https://latex.codecogs.com/gif.latex?\{(Q_j,&space;A_j)\}" /> related to <img src="https://latex.codecogs.com/gif.latex?P" />.

## Result

1. Demonstration with [Jupyter Notebook](https://github.com/holylovenia/question-answer-generation/blob/master/QuestionGeneration/QG_Final.ipynb)

2. Source code for the web version is in [this GitHub repository](https://github.com/agusgun/qag-web). Check it out!

## Installation

1. Export '$STANFORD_PARSER' environment variable using lib path using command `EXPORT STANFORD_PARSER=<ABSOLUTE_PATH>` e.g. `EXPORT STANFORD_PARSER=/home/anonymous/question-answer-generation/QuestionGeneration/lib`
2. If you want the website version please see the link above
3. If the program stuck after loading the model. It's because the port already used. This happened usually after you already run the program twice or several times. Please kill the process that runs on the port using `lsof -i` find java, kill the proces PID.

![result](assets/qag-web.gif)