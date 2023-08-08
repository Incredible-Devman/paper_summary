## CHATGPT-PAPER-READER

<p align="center">
  <img src="./img/robot.png" width="100">
</p>

This repository provides a simple interface that utilizes the `gpt-3.5-turbo` model to read academic papers in PDF format locally.

## Recent Updates
- Cut paper by section titles
- Support handling longer articles and produce summaries for each subsections
- Code refactorization

## How Does This Work

This repo will use ChatGPT to read complete academic papers:

- Splitting a PDF paper into multiple parts for reading and generating a summary of each part. When reading each part, it will refer to the context of the previous part within the token limit.
- Before reading the paper, you can set the questions you are interested in the prompt. This will help ChatGPT focus on the relevant information when reading and summarizing, resulting in better reading performance.
- Answer your question based on all the summaries of all parts of the paper.

By default, the initialized prompt will ask ChatGPT to focus on these points:
- Who are the authors?
- What is the process of the proposed method?
- What is the performance of the proposed method? Please note down its performance metrics.
- What are the baseline models and their performances? Please note down these baseline methods.
- What dataset did this paper use?

These questions are designed for research articles in the field of computer science. After finishing reading the paper, you can ask questions using the `question()` interface.

## Example: Read AlexNet Paper

### Summarize AlexNet
```python
import pickle
from gpt_reader.paper.paper import Paper
from gpt_reader.pdf_reader import PaperReader

reader = PaperReader(openai_key='')
paper = Paper('./alexnet.pdf')
summary = reader.summarize(paper)

# save paper & load
pickle.dump(paper, open('digested_paper.pkl', 'wb'))
paper = pickle.load(open('digested_paper.pkl', 'rb'))
# print summary of a section
print(paper.paper_summaries[4])
```

```
# console print
Beep....Beep....Beep.... Parsing
Beep....Beep....Beep.... I am reading
100%|██████████| 16/16 [02:20<00:00,  8.78s/it]
Bzzzt-klonk... Reading Done, I have built memories for this paper.
```

> reply: ('3 The Architecture\n', "Summary:\nThe authors present the architecture of their network, which contains eight learned layers, including five convolutional and three fully-connected layers. They describe some novel or unusual features of their network's architecture, sorted according to their estimation of their importance.")
  ...


### Ask some questions

```python
session.question(paper, 'Authors of this paper?')
```

> reply: The authors of this paper are Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton from the University of Toronto.


```python
session.question(paper, 'Describe the proposed method in details.')
```

> reply：The proposed method is based on a large, deep convolutional neural network that is trained with stochastic gradient descent using techniques like weight decay, momentum, and dropout for regularization.  The network consists of eight layers (five convolutional and three fully-connected) with the output layer consisting of a 1000-way softmax to produce a distribution over 1000 class labels.  The authors used a ReLU activation function and local response normalization to improve training speed, and overlapping pooling to improve generalization.  They also reduced overfitting by augmenting the training data with techniques like image translations, horizontal reflections, and altering the intensities of the RGB channels in training images using PCA.  The proposed method was trained on subsets of the ImageNet dataset used in the ILSVRC-2010 and ILSVRC-2012 competitions, which contain roughly 1.2 million training images, 50,000 validation images, and 150,000 testing images. By using a large, deep convolutional neural network with various optimization strategies, the proposed method achieved top-5 error rates of 16.4% and 15.3% on the ILSVRC-2012 validation and test sets, respectively, outperforming the previous state-of-the-art results.
> 

```python
session.question(paper, 'What dataset does this paper use for experiment?')
```

> reply: The authors used subsets of the ImageNet dataset used in the ILSVRC-2010 and ILSVRC-2012 competitions for their experiments. These subsets contain approximately 1.2 million training images, 50,000 validation images, and 150,000 testing images.

```python
session.question(paper, 'What metrics does this paper use to evaluate the performace?')
```

> reply: The paper evaluated the performance of their proposed method using Top-1 and Top-5 error rates on the validation and test sets of the ILSVRC-2010 and ILSVRC-2012 image recognition challenges. The Top-1 error rate measures the percentage of images for which the predicted class label is not the correct label, while the Top-5 error rate measures the percentage of images for which the correct label is not among the five most probable labels predicted by the model.


## GUI Interface
![alt](webui.png)
1. Install gradio in your python environment.
```
pip install gradio
```
2. Run gui.py. The link to the web page is output in the console.
```
python gui.py
```
![alt](console.png)
3. Fill in your API_KEY in the appropriate places on the web page and upload the required analysis PDF file. After you wait for the program to finish analyzing, you can switch to the second TAB and ask the program questions about the PDF.

## TODO

- You may exceed the token limit when asking questions.
- More prompt tuning needed to let it outputs stable results.
- Imporve summary accuracies
