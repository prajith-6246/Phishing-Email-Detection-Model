# Phishing Email Detection Model (Mini Project)

A beginner-friendly machine learning project that classifies email text as **Phishing** or **Safe** using Scikit-learn.

## Features
- Loads a labeled CSV dataset of phishing and legitimate email examples
- Analyzes email text with TF-IDF features
- Adds simple URL-related indicators (URL count, suspicious keyword count, and URL presence)
- Trains a Logistic Regression classifier
- Prints accuracy, classification report, and confusion matrix
- Tests a sample email entered by the user

## Requirements
- Python 3.8+
- pandas
- scikit-learn

Install dependencies:

```bash
pip install pandas scikit-learn
```

## Dataset format
Place a CSV file named `emails.csv` in the same folder as the Python script. It must contain these columns:

| email_text | label |
|---|---|
| Your account needs attention. Verify now at http://example.test | Phishing |
| Hello, the meeting is at 3 PM tomorrow. | Safe |

Accepted labels (case-insensitive): `Phishing` and `Safe`.

You can use a suitable public phishing-email dataset, but check its license and ensure the column names/labels match this format. The project does not include a dataset file.

## Run the project

```bash
python phishing_email_detection.py
```

The script:
1. Reads `emails.csv`
2. Splits the data into training and test sets
3. Trains the model
4. Displays accuracy, classification report, and confusion matrix
5. Prompts you to enter an email for prediction

## Evaluation
The displayed accuracy and confusion matrix are based on the held-out test split. Results depend on dataset quality, class balance, and how representative the test data is. High test accuracy does not guarantee real-world detection performance.

## Important limitations
- This is an educational demonstration, not a production email security gateway.
- Text and URL indicators can be misleading; legitimate emails may contain links and phishing messages may avoid obvious keywords.
- Use a sufficiently large, diverse, and correctly labeled dataset.
- Avoid training/testing on duplicate or near-duplicate emails, which can inflate evaluation scores.
- Do not paste private or sensitive email content into public services.

## Project files
- `phishing_email_detection.py` — model training, evaluation, and prediction code
- `README.md` — setup and project guide
- `emails.csv` — user-provided dataset (not included)
