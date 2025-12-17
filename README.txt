# Understanding Human Emotions: The DistilBERT-CNN Way​
Team 12: Chelsea Salyards & Manas Sahoo

## Abstract:

Sentiment analysis has been a challenging topic for AI models to excel at and has been one of the key hurdles in advancing conversational AI adoption. While existing models do almost accurately capture emotions in binary classifications (positive, negative), fine-grained emotion classification is still an area that needs further improvement upon.

In our project we build upon the original research work done to predict fine-grained human emotions, by Stanford Linguistics and Google Research. Our approach uses a multi-input, single-output model using DistilBERT and TextCNN, that combines the strengths of DistilBERT (global context) and TextCNN (local context) to better understand human emotions in texts.

Our primary objective was to use the GoEmotions dataset, the largest human annotated fine-grained emotions dataset, and outperform the baseline benchmarks of an F1-score of 0.46, set by the original research team. Our hybrid model surpassed that with an F1-score of 0.48, precision score of 0.62, and a recall score of 0.43. This demonstrates the effectiveness in using hybrid architecture for identifying wider range of emotions.
This enhanced precision and F1-score finds use in multiple real-world applications where nuanced understanding of emotion is essential. Some of the potential use cases that we can think of are next-gen AI therapists providing more empathetic and context-aware responses and improved customer service bots for customer sentiment assessment. Through this project we do provide a example for further research work to be done in sentiment analysis through hybrid architecture.

## Run instructions:

Ensure supporting files in dataset.zip are downloaded and take note of their location.
When running the Jupyter Notebook, review the Configuration flags block and ensure the following parameters are set to the appropriate file locations on your machine:
* emotion_file
* sentiment_file
* output_dir
* data_dir
* train_fname
* dev_fname
* test_fname

data_dir is the location of the .csv dataset files.
output_dir does not need to exist ahead of time, but needs to be in a writeable location.

## Recommended module versions:
Tensorflow: 2.19.0
transformers: 4.55.2
safetensors: 0.6.2
pandas: 2.2.2
matplotlab: 3.10.0
Keras: 3.10.0
sklearn: 1.6.1
numpy: 2.0.2
requests: 2.32.3
seaborn: 0.13.2

## References:
* Demszky, Dorottya, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. ‘GoEmotions: A Dataset of Fine-Grained Emotions’, 5 2020. http://arxiv.org/abs/2005.00547.
* Sanh, Victor, Lysandre Debut, Julien Chaumond, and Thomas Wolf. ‘DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter’, 2019. https://github.com/huggingface/transformers.
* Abas, Ahmed R., Ibrahim Elhenawy, Mahinda Zidan, and Mahmoud Othman. ‘Bert-Cnn: A Deep Learning Model for Detecting Emotions from Text’. Computers, Materials and Continua 71 (2022): 2943–61. https://doi.org/10.32604/cmc.2022.021671.
* Guo, Yuan, and Hongmei Li. ‘Research on Sentiment Analysis of Online Course Evaluation Based on EN- BERT-CNN’. In Proceedings of the 2024 8th International Conference on Electronic Information Technology and Computer Engineering, 65–68. ACM, 10 2024. https://doi.org/10.1145/3711129.3711142.
* Duong, Chi Thang, Remi Lebret, and Karl Abererécole. ‘Multimodal Classification for Analysing Social Media’, 2017.
* Tan, Kian Long, Chin Poo Lee, Kalaiarasi Sonai Muthu Anbananthen, and Kian Ming Lim. ‘RoBERTa-LSTM: A Hybrid Model for Sentiment Analysis With Transformer and Recurrent Neural Network’. IEEE Access 10 (2022): 21517–25. https://doi.org/10.1109/ACCESS.2022.3152828.
* Kokab, Sayyida Tabinda, Sohail Asghar, and Shehneela Naz. ‘Transformer-Based Deep Learning Models for the Sentiment Analysis of Social Media Data’. Array 14 (7 2022). https://doi.org/10.1016/j.array.2022.100157.
* Wang, Shuo, Yuqi Lu, Ruijue Luo, Boxuan Li, Yingjie Zhang, and Liang Xiao. ‘A BERT-Based Sentiment Analysis Model for Depressive Text’. In Proceedings of the 2025 International Conference on Generative Artificial Intelligence and Digital Media, 76–83. ACM, 3 2025. https://doi.org/10.1145/3734921.3734934.
* Kim, Yoon. ‘Convolutional Neural Networks for Sentence Classification’, 8 2014. http://arxiv.org/abs/1408.5882.
* GoEmotions data set: https://www.kaggle.com/datasets/debarshichanda/goemotions Downloaded June 10, 2025
* GoEmotions source code: https://github.com/google-research/google-research/tree/master/goemotions Downloaded June 10, 2025
* TextCNN: https://github.com/delldu/TextCNN Downloaded July 11, 2025
* HuggingFace documentation: https://huggingface.co/docs/transformers/en/model_doc/distilbert
* Keras documentation: https://www.tensorflow.org/tutorials/estimator/keras_model_to_estimator https://www.tensorflow.org/guide/migrate/migrating_estimator
* F1 Score in Machine Learning. 2025: https://www.futurense.com/Uni-Blog/F1-Score-Machine-Learning.
* Classification Metrics Guide: https://www.evidentlyai.com/classification-metrics/accuracy-precision-recall
* GoEmotions emotions via emojis: https://neurohive.io/en/datasets/go-emotions-google-ai-dataset-for-sentiment-analysis/#:~:text=31%20October%202021,as%20improve%20customer%20support%20services
* Metrics for Sentiment Analysis: https://www.getfocal.co/post/top-7-metrics-to-evaluate-sentiment-analysis-models