# A Dive into Natural Language Processing

![](https://i.imgur.com/GTH36GV.png)

⭐ [градиент обреченный](https://t.me/doomgrad) — телеграм канал про машинное обучение.

## 1. Лекция. Обзор современного машинного обучения

Расскажу про основные домены в DS, в каком состоянии они сейчас находятся и какие задачи сейчас актуальны на рынке (в ритейле, банках, промышленности, ИБ и т.д.).

_Upd. 09.10.2021_

🎬 https://www.youtube.com/watch?v=LT8QJcOFrwo

🗨️ https://docs.google.com/presentation/d/1YFy-Ia7qwkiQp8nlHk7__JIFeB-dAnzI

**✨ Основные виды данных**

- Табличные данные и врменные ряды
- Видео и звук
- Картинки
- Текстовые данные

**✨ Типичные задачи ML**

- Forecasting
- Anomaly detection
- Voice recognition
- Text to Speech
- Video captioning
- Object detection
- Segmentation
- Style transfer
- Image generation
- Noise reduction
- Super Resolution
- Machine translation
- NER
- Relation extraction
- Question answering
- Classification (spam, sentiment, etc.)
- Summarization
- Topic modelling

#### Open Data Science community

- https://ods.ai

## 2. Семинар. Делаем семантический поиск

Рассмотрим основные инструменты, подходы и терминологию, которые используются в ML/DS. Разберем, как переводить текстовые данные в векторное пространство. Сделаем простой семантический поиск по текстам. Начнем работать с Colab.

_Update 16.10.2021_

🎬 https://www.youtube.com/watch?v=VJTXBDHpsus

⚡ https://colab.research.google.com/drive/1sBavnRdQTR7NDZDgLwv6_yVnu_UY_PL2?usp=sharing

**✨ Понятия и термины**

- Tokenization
- Lemmatization
- Stemming
- Distributional semantics
- Embedding
- Word2Vec
- GloVe
- fastText

## 3. Лекция. Векторное представление текста. Embeddings.

Базовые подходы и техники, с которых начинается решение прикладных задач и часто же ими и заканчивается. Разберем основные алгоритмы. Статистические и нейросетевые подходы по переводу текстов в векторное пространство. Word2Vec.

_Update 24.10.2021_

🎬 https://www.youtube.com/watch?v=OV_QM_BuAhU

🗨️ https://docs.google.com/presentation/d/162aedK5-nubUV-Z59zQ5zMW5nnLyB8Gu

**✨ Понятия и термины**

- One-hot encoding
- Bag of Words
- N-grams
- TF-IDF
- Distributional semantics
- Pointwise mutual information
- Matrix factorization
- SVD
- Word2Vec
- Subsampling
- Negative sampling

## 4. Семинар. Классификация текстов. Transfer learning

_Upd. 30.10.2021_

Применим базовые подходы для решения задач. Узнаем, что такое transfer learning и как начать просто использовать предобученные модели через _huggingface_.

🎬 https://www.youtube.com/watch?v=uRAsurPHycw

⚡ https://colab.research.google.com/drive/1xtkx4pj3v7lNKXJvD63nu4YGn7YFlAlC?usp=sharing

**✨ Понятия и термины**

- Neural nets
- Text classification
- Metrics
- Transfer learning
- Pretrainig
- Huggingface
- Interview questions

**⭐️ Полезные ссылки**

- [Теоретические вопросы по анализу данных для общего развития и подготовки к интервью](https://github.com/alexeygrigorev/data-science-interviews/blob/master/theory.md)
- [Видео от Deep Learning School (МФТИ) про полносвязные сети и механизм обучения](https://www.youtube.com/watch?v=O0nGKKFyYT4)
- [Заметка от Евгения Соколова про метрики классификации](https://github.com/esokolov/ml-course-msu/blob/master/ML15/lecture-notes/Sem05_metrics.pdf)

## 5. Лекция. Переломный момент в ML

_Upd. 10.11.2021_

Как повлиял механизм внимания на развитие ML. Расскажу про трансформеры, которые сейчас являются SOTA во многих областях.

🎬 https://www.youtube.com/watch?v=6b0MXyHbILs

🗨️ https://docs.google.com/presentation/d/1rgKZaypYtjunoptDZkvoc4JDXCcNgtlU

**✨ Понятия и термины**

- Механизм внимания (Attention, Self-attention)
- RNN
- LSTM
- Encoder
- Decoder
- Transformer

## 6. Семинар. Знакомимся с PyTorch и PyTorch Lightning. Пишем первую нейросеть.

Поговорим, что такое PyTorch и для чего он нужен. Потренируем сеть на MNIST'е. Отрефакторим в PyTorch Lightning, чтобы было проще работать с моделью.

🎬 https://www.youtube.com/watch?v=Oc-DX3xwyFA

⚡ https://colab.research.google.com/drive/1K1hz93ceM926vyEs0Ouxbizu458GDHd0

**✨ Понятия и термины**

- PyTorch
- Tensor
- Optimizer
- Loss function
- PyTorch Lightning
- DataModule
- TensorBoard

## 7. Лекция. Машинный перевод

_Update 20.11.2021_

Машинный перевод — движущая сила NLP. Поговорим про его развитие, про современные модели, про проблемы, связанные с их обучением.

🎬 https://youtu.be/8JqWdACYKns

🗨️ https://docs.google.com/presentation/d/1QtWcCkZQ6RqgsXohPy3Pm-t3iyIwAj5l

**✨ Понятия и термины**

- Machine translation
- BLEU, NIST, METEOR
- Джорджтаунский эксперимент
- ALPAC report
- Parallel corpora
- RBMT. Rule-based machine translation
- EBMT. Example-based machine translation
- SMT. Statistical machine translation
- NMT. Neural machine translation

**⭐️ Ссылки**

- [Awesome Machine Translation](https://github.com/maidis/awesome-machine-translation)
- [OpenNMT](https://opennmt.net/)
- [Лекция от МФТИ по машинному переводу](https://www.youtube.com/watch?v=6HibilFua-U)
- [The first-ever multilingual model to win WMT](https://ai.facebook.com/blog/the-first-ever-multilingual-model-to-win-wmt-beating-out-bilingual-models/)
- [mT5](https://github.com/google-research/multilingual-t5)
- [ruT5, ruRoBERTa, ruBERT](https://habr.com/ru/company/sberbank/blog/567776/)


## 8. Семинар. Машинный перевод

_Upd. 26.11.2021_

Сделаем русско-английский переводчик. Зафайнтюним модель mT5, которую можно адаптировать под множество прикладных задач. Про это тоже поговорим.

🎬 https://www.youtube.com/watch?v=vVnYib1MiYY&t=732s

⚡ https://colab.research.google.com/drive/1KEGej1rIWBGpljZSMCjj7_MFv3c7DG3L

**✨ Понятия и термины**

- Машинный перевод
- T5
- mT5
- Fine-tuning
- Multitask model training

---

_tg: [@averkij](https://t.me/averkij)_ _[@doomgrad](https://t.me/doomgrad)_
