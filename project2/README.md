**Description**

X (formerly known as Twitter) has become a major platform for users to share opinions, emotions, and information worldwide. Millions of posts on X cover a large range of topics and events, reflecting public sentiment and offering valuable insights for public opinion monitoring. 

BERT, with its powerful contextual understanding capabilities, has made significant strides in natural language processing tasks, enabling more accurate detection of sentiment nuances. Meanwhile, LoRA, an optimization technique, improves model efficiency by reducing training parameters without sacrificing sentiment analysis accuracy. This innovative combination not only enhances the quality of sentiment classification but also makes processing large-scale X data more efficient and feasible. Thus, this project aims to provide a novel solution for post sentiment analysis using the synergy of BERT and LoRA, enabling users to more accurately understand and predict emotional dynamics on social media.

The dataset can be accessed directly from Kaggle [Twitter Sentiment Analysis
](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

**Project Steps**

1. Data preparation and EDA
2. Save the best performing model as  `model_lora`
3. Load model to script `predict.py`, Serving it via a web service with Flask.
4. Containerization with Docker
