# 🚗 AutoBERT
# Fine-Tuned DistilBERT for Automotive Reviews Classification

## 📌 Project Overview
This project fine-tunes **DistilBERT** (a transformer-based language model) for **sentiment classification of Amazon automotive reviews**.  
It was developed as part of **Natural Language Processing (NLP) Assignment #4** by **Ofek Cohen** and **Omer Blau**.

The goal was to adapt a HuggingFace tutorial for sequence classification to a custom dataset (`Automotive.train.json` and `Automotive.test.json`).  

---

## ⚙️ Tasks
### 1. Fine-Tuning DistilBERT
- Dataset: Automotive reviews (train/test).  
- Model: `distilbert-base-uncased`.  
- Training setup followed HuggingFace’s **transformers** library guidelines.  
- Hyperparameters:  
  ```python
  TrainingArguments(
      output_dir="./model",
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      num_train_epochs=2,
      weight_decay=0.01,
      push_to_hub=False,
      report_to="none"
  )
  ```

- Final model was evaluated on the **test set**, yielding higher accuracy compared to TF-IDF + Logistic Regression baseline.

### 2. Reflections on AI Tools
Alongside the technical part, we wrote a short essay (Hebrew) reflecting on the **advantages and challenges of rapidly developing AI tools** from perspectives of:  
- **Society** (accessibility, fake news risk)  
- **Academia** (better tools, constant adaptation)  
- **Personal life** (recommendation systems vs. privacy concerns)  

---

## 📝 Dataset
- **Automotive.train.json** – training split (Amazon automotive reviews).  
- **Automotive.test.json** – test split for evaluation.  
Each review includes:  
- `reviewText` – the text of the review  
- `overall` – rating (label)

---

## 🚀 How to Run
### Requirements
Install dependencies:
```bash
pip install torch transformers datasets scikit-learn
```

### Training
Run the Jupyter notebook:
```bash
Ofek_and_Omer_sequence_classification.ipynb
```

This will:  
1. Load and tokenize the dataset.  
2. Fine-tune DistilBERT.  
3. Evaluate on the test set.  

### Output
- Fine-tuned model weights saved locally in `./model/`.  
- Printed **test accuracy** in the final cell of the notebook.  

---

## 📊 Results
- Fine-tuned DistilBERT significantly outperformed the baseline logistic regression model.  
- Final accuracy (test set): **(see notebook output)**.  

---

## 👩‍💻 Authors
- **Ofek Cohen**  
- **Omer Blau**
