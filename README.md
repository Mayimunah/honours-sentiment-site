# Honours Sentiment Explorer

A Streamlit reconstruction for Mayimunah Nagayi's portfolio.

## Run on Windows
Install Python 3.12. Extract this ZIP, open a terminal inside the extracted folder, then run:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m streamlit run app.py
```

## Deploy
1. Create a GitHub repository such as honours-sentiment-explorer.
2. Upload the CONTENTS of this folder, including models and .streamlit. app.py should be at the repository root. Do not upload the ZIP or a virtual environment.
3. Sign in at https://share.streamlit.io using GitHub.
4. Create an app from your repository, select the main branch and app.py.
5. Under advanced settings select Python 3.12, then deploy.
6. Open the resulting .streamlit.app URL, test text input and CSV upload, and add that URL to your portfolio.

Official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## What is included
Text entry, CSV upload with column selection, AP and K-means assignments, cluster counts and downloadable results. Uploads are processed in memory. No credentials or external model API are required. Up to 500 texts and 5 MB per CSV.

## Research fidelity
Source: https://github.com/Mayimunah/Affinity-progation-sentiment-analysis
Source commit: 5c7d2dfad6c47928ab4873ee441e9627f19de345

Numeric parameters are exported from the repository's original scikit-learn 1.2.2 pickle files into NPZ and JSON to avoid dependence on old pickle internals. TF-IDF uses the saved vocabulary/IDF and default original settings. AP uses the original SVD then nearest exemplar; K-means uses nearest centre in TF-IDF space. No retraining occurs. This differs from the original AP testing notebook's fit_predict on each new batch. Agglomerative labels are not a reusable fitted classifier and are not used here. No verified cluster-to-sentiment mapping was recovered, so the app deliberately displays cluster IDs. Do not describe this build as a validated positive/negative/neutral classifier.

Cleaning follows the AP notebook: strip nonletters, remove lowercase 'user', remove English NLTK stopwords, lowercase. Whitespace splitting is equivalent for this letters-only input. The K-means notebook had a different mention-cleaning expression; this build uses one documented cleaning path for both. Current NLTK stopwords are bundled, so exact historical list equivalence has not been established. Known limitations include loss of negation, English-only vocabulary, and the saved AP model's partial training state. The original full corpus is not bundled.

This first build is a recovered-model exploration app. Verifying the original sentiment mapping and restoring any additional demo behaviours remains future work.
