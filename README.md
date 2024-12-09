# RAG Q&A Chatbot with Llama2 or Flan-T5
## **Introduction**
This repository hosts a Retrieval-Augmented Generation (RAG) Question-Answering Chatbot powered by state-of-the-art language models such as Llama2 or Flan-T5.
The chatbot first retrieves relevant information from Wikipedia articles and then uses a language model to generate contextually accurate answers. This approach is ideal for applications that require precise and well-informed responses, such as customer support, educational tools, and research assistants.

## **Installation**

This project is run on Kaggle instead of Google Colab. Before running the code, ensure that the following libraries are pre-installed:

- **Torch**: For deep learning support and model training.
- **Transformers**: For utilizing pre-trained models and tokenizers.
- **Tqdm**: For displaying progress bars in loops.
- **Pandas**: For data manipulation and analysis.
- **LangChain**: For building modular language model applications.
- **BitsAndBytes**: For efficient model parameter handling and memory optimization.
- **PEFT (Parameter-Efficient Fine-Tuning)**: For lightweight model fine-tuning.
- **Accelerate**: For easy multi-GPU and distributed training.

To install these libraries, you can use the following command:

```python
!pip install torch transformers tqdm pandas langchain bitsandbytes peft accelerate
```

## **Data Sources**
This project's Wikipedia articles and Q&A data were downloaded from [Question-Answer Dataset](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset).

The dataset includes three question files, one for each year: S08, S09, and S10. I used only S08 for faster results.
The columns in this file are as follows: ArticleTitle, Question, Answer, DifficultyFromQuestioner, DifficultyFromAnswerer, ArticleFile.

To evaluate the model the data was split into 80% (481 questions) for training and 20% (121 questions) for validation.

## **Training Results**
I used the FAISS vector store because creating embeddings takes ~3 minutes, while it will take ~35 minutes with the Chroma vector store. Furthermore, Flan-T5 was fine-tuned using the PEFT method LoRA. 
Finally, I achieved a **67% ROUGE Recall Score** with Flan-T5 and **71% ROUGE Recall Score** with larger models like wizardLM or llama2-7b.

## **Future Work**
I didn't have time to try 'flan-t5-base-squad2' which is fine-tuned for Extractive QA. Also, it's worth testing different embedding models and pipeline parameters (temperature, top_p, penalty).
