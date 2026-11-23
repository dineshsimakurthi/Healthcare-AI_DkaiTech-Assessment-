# Healthcare AI Prototype - ML Prediction + RAG Q&A

A small demonstration prototype with two parts:

1. **Machine Learning** - predicts diabetes from patient measurements (scikit-learn).
2. **RAG** - answers healthcare questions from 8 documents using a LangChain **LCEL** chain and **ChromaDB**, and shows its sources.
3. **Interface (optional)** - a Streamlit app combining both.

> Demonstration only. Not medical advice.

## Project structure
```
healthcare-ai/
├── data/diabetes.csv            # Pima Indians Diabetes dataset (768 rows)
├── docs/                        # 8 healthcare text documents used by RAG
├── src/
│   ├── train_model.py           # Part 1: clean, split, train 2 models, metrics, save
│   ├── predict.py               # Part 1: prediction function
│   ├── build_index.py           # Part 2: load -> clean -> chunk -> embed -> Chroma
│   └── rag_chain.py             # Part 2: LCEL chain + sources + "not found" handling
├── app.py                       # Part 3: Streamlit app
├── models/                      # saved model + metrics.json (created by train_model.py)
└── requirements.txt
```

## Setup
```bash
python -m venv venv && source venv/bin/activate     # optional
pip install -r requirements.txt
```
Google Colab: upload the folder, then run the same `pip install` in a cell and use `!python ...` for each command.

**LLM key** (only needed for RAG answers; the ML part needs no key):
```bash
export OPENAI_API_KEY="sk-..."            # default provider (gpt-4o-mini)
# Alternatives:
# export LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=...   (pip install langchain-anthropic)
# export LLM_PROVIDER=ollama                            (free & local; pip install langchain-ollama)
```

## Run
```bash
python src/train_model.py       # Part 1: train + evaluate + save model
python src/predict.py           # try the prediction function
python src/build_index.py       # Part 2: build the Chroma vector store (run once)
cd src && python rag_chain.py   # ask questions in the terminal
streamlit run app.py            # Part 3: web interface
```

## Part 1 - Machine learning
**Dataset:** Pima Indians Diabetes (public; 768 patients, 8 features, `Outcome` = diabetic or not).
Download link: https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv

**Pipeline**
- *Cleaning:* drop duplicates. In `Glucose, BloodPressure, SkinThickness, Insulin, BMI` a value of 0 is physically impossible, so it is treated as **missing**.
- *Missing values:* filled with the **median** (`SimpleImputer`) inside a scikit-learn `Pipeline`, so it is learned from training data only (no data leakage).
- *Split:* 80% train / 20% test, stratified (keeps the 65/35 class ratio).
- *Models:* Logistic Regression (with scaling) and Random Forest.
- *Metrics:* accuracy, precision, recall, F1, ROC-AUC (saved to `models/metrics.json`).

**Results (test set, 154 patients)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 | **0.813** |
| Random Forest | 0.740 | 0.659 | 0.537 | 0.592 | 0.807 |

**Which model is best and why?** The two are very close. Random Forest has slightly higher accuracy/precision/recall/F1 at the default 0.5 threshold, but **Logistic Regression has the best ROC-AUC (0.813)**. ROC-AUC measures how well the model ranks diabetic above non-diabetic patients regardless of threshold, which matters here because the classes are imbalanced (~35% diabetic) and a screening tool would tune its threshold. Logistic Regression is also simpler and easier to explain to clinicians, so it is saved as the final model. (With only 154 test rows, the differences are within noise.)

**Prediction function:** `predict_risk(patient_dict)` in `src/predict.py` returns probability, risk level and prediction.

## Part 2 - RAG system
**Documents:** 8 short text files in `docs/` covering diabetes (overview, symptoms, diagnosis, risk factors, lifestyle, complications) and hypertension (risk factors, lifestyle, when to see a doctor). They are plain-language summaries of public guidance from WHO, CDC, NIDDK and the American Heart Association. You can add more `.txt` files and re-run `build_index.py`.

**Flow**
1. **Load & clean** the documents (`build_index.py`)
2. **Chunk** with `RecursiveCharacterTextSplitter` (500 chars, 80 overlap)
3. **Embed** with `sentence-transformers/all-MiniLM-L6-v2` (free, local)
4. **Store** in **ChromaDB** (`chroma_db/`)
5. **Retrieve** top-4 chunks for the question
6. **Generate** the answer with an LLM using the LCEL chain
7. **Show sources**; if the context has no answer the model must reply *"I could not find this information in the provided documents."* and no sources are shown

**The LCEL chain** (`rag_chain.py`)
```python
answer_chain = PROMPT | llm | StrOutputParser()

chain = (
    RunnableParallel(docs=retriever, question=RunnablePassthrough())
    .assign(answer=RunnableLambda(build_prompt_inputs) | answer_chain)
)
```
The `|` pipe passes the output of each step to the next one.

## External libraries, APIs and pretrained models
- scikit-learn, pandas, numpy, joblib
- LangChain (`langchain-core`, `langchain-chroma`, `langchain-huggingface`, `langchain-text-splitters`), ChromaDB
- Pretrained embedding model: `sentence-transformers/all-MiniLM-L6-v2` (Hugging Face)
- LLM API: OpenAI `gpt-4o-mini` (or Anthropic / Ollama via `LLM_PROVIDER`)
- Streamlit (Part 3)

## Limitations and future improvements
- Small dataset (768 patients, one population); results are not generalisable and recall is modest (~0.5), so many diabetic cases are missed.
- Missing values (especially Insulin and SkinThickness) were imputed with the median, which adds noise.
- No hyper-parameter tuning or cross-validation; adding both would give more reliable estimates.
- The RAG corpus is tiny (8 short documents) and written as summaries; for real use, load official PDFs (WHO, CDC, ADA) and keep them updated.
- Retrieval is plain similarity search; hybrid search, re-ranking and a relevance-score threshold would reduce wrong or off-topic answers.
- LLM answers can still be imperfect; there is no automatic evaluation of RAG quality (e.g. RAGAS).
- Not a medical device and not clinically validated; it must not be used for diagnosis.
