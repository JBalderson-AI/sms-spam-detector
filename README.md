# Module 21: SMS Spam Detector

## Project Overview
This project implements an **SMS Spam Detector** using **machine learning**. The model is trained to classify text messages as either **"spam"** or **"not spam"** using **TF-IDF vectorization** and a **Linear Support Vector Classifier (SVC)**. The project is divided into two Jupyter notebooks:

1. **sms_text_classification_solution.ipynb** → Trains the model and saves it.
2. **gradio_sms_text_classification.ipynb** → Loads the trained model and provides a **Gradio-based web interface** for user input.

---

## Technologies & Techniques
This project leverages the following technologies and techniques:
- **Python**: Core programming language for data processing and machine learning.
- **Pandas**: Data manipulation and preprocessing.
- **Scikit-learn**: Machine learning library used for TF-IDF vectorization and model training.
- **TfidfVectorizer**: Converts text into numerical features based on Term Frequency-Inverse Document Frequency.
- **LinearSVC**: A linear Support Vector Machine classifier for text classification.
- **Gradio**: Creates an interactive web interface for user testing.
- **Joblib**: Saves and loads the trained model to avoid retraining.

---

## Project Structure
```
├── sms_text_classification_solution.ipynb  # Notebook to train & save model
├── gradio_sms_text_classification.ipynb   # Notebook to load model & create Gradio app
├── SMSSpamCollection.csv                   # Dataset (SMS messages & labels)
├── sms_spam_model.pkl                      # Saved trained model (Generated after training)
├── README.md                               # Project documentation
```

---

## Installation & Setup
### 1. Install Required Libraries

### 2. Train the Model (First Notebook)
1. Open **`sms_text_classification_solution.ipynb`**.
2. Run all cells to **train the spam detection model**.
3. The trained model will be saved as **`sms_spam_model.pkl`**.

### 3. Launch the Gradio App (Second Notebook)
1. Open **`gradio_sms_text_classification.ipynb`**.
2. Run all cells to **load the trained model**.
3. The Gradio web interface will launch.

---

## How to Use the Spam Detector
1. **Enter a text message** in the Gradio interface.
2. **Click "Submit"** to classify the message.
3. The app will return:
   - **"Not Spam"** → If the message is safe.
   - **"Spam"** → If the message is classified as spam.

### Example Inputs & Expected Outputs
| Input Message                                    | Expected Output |
|------------------------------------------------|----------------|
| `"Win a free iPhone now!"`                    | Spam           |
| `"Hey, are we still on for coffee?"`          | Not Spam       |
| `"Congratulations! You've won $1000!"`       | Spam           |
| `"Let's meet at 5 PM."`                       | Not Spam       |

---

## Model Performance
After training, the model is evaluated using accuracy metrics. Run the following command in **`sms_text_classification_solution.ipynb`** to check its performance:
```python
print('Train Accuracy:', text_clf.score(X_train, y_train))
print('Test Accuracy:', text_clf.score(X_test, y_test))
```

---

## Troubleshooting
- **Gradio app is not launching?**
  ```sh
  pip install gradio --upgrade
  ```
- **Getting inaccurate predictions?**
  - Ensure the dataset is correctly loaded.
  - Check that `sms_spam_model.pkl` is being used instead of retraining the model.
  - Try adjusting `TfidfVectorizer` parameters to reduce overfitting.

---

## License
This project is licensed under the **MIT License**.

---

## Credits
Jill Balderson - JBalderson-AI





Built with **Python, Scikit-learn, and Gradio**

