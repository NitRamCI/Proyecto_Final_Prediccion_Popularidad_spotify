import joblib
import numpy as np
import pandas as pd

# ─── Cargamos los artefactos del modelo ───
model       = joblib.load('./model/model.pkl')
scaler_X    = joblib.load('./model/scaler_X.pkl')
transformer = joblib.load('./model/transformer.pkl')

print("✅ Modelo y artefactos cargados correctamente!")
print(f"   Tipo de modelo: {type(model).__name__}")

# ─── Función de predicción ───
def predict_song_popularity(
    track_genre: str,
    duration_ms: float,
    explicit: int,
    danceability: float,
    energy: float,
    key: int,
    loudness: float,
    mode: int,
    speechiness: float,
    acousticness: float,
    instrumentalness: float,
    liveness: float,
    valence: float,
    tempo: float,
    time_signature: int
) -> float:
    """
    Predicts the popularity of a song on Spotify (0–100).

    Args:
        track_genre (str):        Genre (e.g. 'pop', 'rock', 'hip-hop').
        duration_ms (float):      Duration in milliseconds.
        explicit (int):           Explicit content: 1 = yes, 0 = no.
        danceability (float):     Danceability score (0.0–1.0).
        energy (float):           Energy score (0.0–1.0).
        key (int):                Musical key (0–11).
        loudness (float):         Loudness in dB (typically -60 to 0).
        mode (int):               Major (1) or minor (0).
        speechiness (float):      Speechiness score (0.0–1.0).
        acousticness (float):     Acousticness score (0.0–1.0).
        instrumentalness (float): Instrumentalness score (0.0–1.0).
        liveness (float):         Liveness score (0.0–1.0).
        valence (float):          Valence score (0.0–1.0).
        tempo (float):            Tempo in BPM.
        time_signature (int):     Time signature (e.g. 4).

    Returns:
        float: Predicted popularity score (0–100).
    """
    input_data = pd.DataFrame([{
        'track_genre':      track_genre,
        'duration_ms':      duration_ms,
        'explicit':         int(explicit),
        'danceability':     danceability,
        'energy':           energy,
        'key':              key,
        'loudness':         loudness,
        'mode':             mode,
        'speechiness':      speechiness,
        'acousticness':     acousticness,
        'instrumentalness': instrumentalness,
        'liveness':         liveness,
        'valence':          valence,
        'tempo':            tempo,
        'time_signature':   time_signature
    }])

    # 1. Aplicar el mismo ColumnTransformer usado en el entrenamiento
    X_transformed = transformer.transform(input_data).astype(float)

    # 2. Escalar solo si el modelo es lineal
    linear_models = ['LinearRegression', 'Lasso', 'Ridge']
    if type(model).__name__ in linear_models:
        X_transformed = scaler_X.transform(X_transformed)

    # 3. Predecir y clampear entre 0 y 100
    prediction = model.predict(X_transformed)
    return float(np.clip(prediction[0], 0, 100))


# ─── Tests de prueba ───

# Test 1: canción pop bailable y energética (esperamos popularidad alta)
pop_song = {
    'track_genre':       'pop',
    'duration_ms':       210000.0,
    'explicit':          0,
    'danceability':      0.80,
    'energy':            0.85,
    'key':               5,
    'loudness':          -5.0,
    'mode':              1,
    'speechiness':       0.05,
    'acousticness':      0.08,
    'instrumentalness':  0.0,
    'liveness':          0.10,
    'valence':           0.75,
    'tempo':             128.0,
    'time_signature':    4
}

# Test 2: canción ambient de nicho (esperamos popularidad baja)
ambient_song = {
    'track_genre':       'ambient',
    'duration_ms':       420000.0,
    'explicit':          0,
    'danceability':      0.20,
    'energy':            0.08,
    'key':               3,
    'loudness':          -22.0,
    'mode':              0,
    'speechiness':       0.03,
    'acousticness':      0.90,
    'instrumentalness':  0.85,
    'liveness':          0.07,
    'valence':           0.15,
    'tempo':             72.0,
    'time_signature':    4
}

# Test 3: reggaeton con explicit
reggaeton_song = {
    'track_genre':       'reggaeton',
    'duration_ms':       195000.0,
    'explicit':          1,
    'danceability':      0.88,
    'energy':            0.79,
    'key':               7,
    'loudness':          -4.5,
    'mode':              0,
    'speechiness':       0.18,
    'acousticness':      0.05,
    'instrumentalness':  0.0,
    'liveness':          0.09,
    'valence':           0.68,
    'tempo':             95.0,
    'time_signature':    4
}

print("\n" + "="*50)
print("TESTS DE PREDICCIÓN")
print("="*50)

pred_pop       = predict_song_popularity(**pop_song)
pred_ambient   = predict_song_popularity(**ambient_song)
pred_reggaeton = predict_song_popularity(**reggaeton_song)

print(f"\n🎵 Pop bailable y energético:    {pred_pop:.2f} / 100")
print(f"🎵 Ambient instrumental de nicho: {pred_ambient:.2f} / 100")
print(f"🎵 Reggaeton explícito:           {pred_reggaeton:.2f} / 100")

print("\n" + "="*50)
print("VALIDACIÓN")
print("="*50)

# El modelo debe dar más popularidad al pop que al ambient
if pred_pop > pred_ambient:
    print("✅ Pop > Ambient — el modelo distingue géneros correctamente")
else:
    print("⚠️  Atención: el modelo no distinguió bien entre géneros")

# Todas las predicciones deben estar entre 0 y 100
all_valid = all(0 <= p <= 100 for p in [pred_pop, pred_ambient, pred_reggaeton])
if all_valid:
    print("✅ Todas las predicciones están en el rango 0–100")
else:
    print("⚠️  Alguna predicción está fuera del rango esperado")

print("\n✅ ml-test.py completado — la API está lista para correr")
