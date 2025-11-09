import joblib

model_rf = joblib.load('saved_models/random_forest_audio_multilabel.joblib')
pca = joblib.load('saved_models/pca_selected_audio_multilabel.joblib')
scaler = joblib.load('saved_models/scaler_audio_multilabel.joblib')


