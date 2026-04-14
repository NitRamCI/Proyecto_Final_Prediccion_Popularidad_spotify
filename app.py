from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow

app = Flask(__name__)

#### CONFIGURACION DE SQLALCHEMY ####
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db_spotify.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

### TABLA: almacena cada canción consultada y su predicción de popularidad
class Song(db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    track_genre      = db.Column(db.String(100), nullable=False)
    duration_ms      = db.Column(db.Float, nullable=False)
    explicit         = db.Column(db.Integer, nullable=False)   # 0 o 1
    danceability     = db.Column(db.Float, nullable=False)
    energy           = db.Column(db.Float, nullable=False)
    key              = db.Column(db.Integer, nullable=False)
    loudness         = db.Column(db.Float, nullable=False)
    mode             = db.Column(db.Integer, nullable=False)
    speechiness      = db.Column(db.Float, nullable=False)
    acousticness     = db.Column(db.Float, nullable=False)
    instrumentalness = db.Column(db.Float, nullable=False)
    liveness         = db.Column(db.Float, nullable=False)
    valence          = db.Column(db.Float, nullable=False)
    tempo            = db.Column(db.Float, nullable=False)
    time_signature   = db.Column(db.Integer, nullable=False)
    popularity       = db.Column(db.Float, nullable=True)      # predicción

### SCHEMA: serializa los objetos Song a JSON
ma = Marshmallow(app)

class SongSchema(ma.Schema):
    id               = ma.Integer()
    track_genre      = ma.Str()
    duration_ms      = ma.Float()
    explicit         = ma.Integer()
    danceability     = ma.Float()
    energy           = ma.Float()
    key              = ma.Integer()
    loudness         = ma.Float()
    mode             = ma.Integer()
    speechiness      = ma.Float()
    acousticness     = ma.Float()
    instrumentalness = ma.Float()
    liveness         = ma.Float()
    valence          = ma.Float()
    tempo            = ma.Float()
    time_signature   = ma.Integer()
    popularity       = ma.Float()

## Creamos las tablas en la base de datos
db.create_all()
print('Tablas creadas correctamente')

##### MODELO ML #####
import joblib
import numpy as np
import pandas as pd

# Cargamos los artefactos generados en el notebook de entrenamiento
model       = joblib.load('./model/model.pkl')
scaler_X    = joblib.load('./model/scaler_X.pkl')
transformer = joblib.load('./model/transformer.pkl')

# Nombre del mejor modelo (para saber si necesita escalado)
LINEAR_MODELS = ['Linear Regression', 'Lasso', 'Ridge']

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
    Recibe las características de una canción,
    aplica el mismo pipeline del notebook y retorna
    la popularidad predicha (0–100).
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

    # 1. Aplicar OneHotEncoder (mismo transformer del notebook)
    X_transformed = transformer.transform(input_data).astype(float)

    # 2. Escalar solo si el modelo es lineal
    model_type = type(model).__name__
    if any(lm.replace(' ', '') in model_type for lm in ['LinearRegression', 'Lasso', 'Ridge']):
        X_transformed = scaler_X.transform(X_transformed)

    # 3. Predecir y clampear entre 0 y 100
    prediction = model.predict(X_transformed)
    return float(np.clip(prediction[0], 0, 100))


##### RUTAS #####

@app.route('/')
def index():
    return render_template('index.html')


# POST /song — predice la popularidad y guarda el registro
@app.route('/song', methods=['POST'])
def create_song():
    data = request.json

    track_genre      = data.get('track_genre')
    duration_ms      = data.get('duration_ms')
    explicit         = data.get('explicit', 0)
    danceability     = data.get('danceability')
    energy           = data.get('energy')
    key              = data.get('key')
    loudness         = data.get('loudness')
    mode             = data.get('mode')
    speechiness      = data.get('speechiness')
    acousticness     = data.get('acousticness')
    instrumentalness = data.get('instrumentalness')
    liveness         = data.get('liveness')
    valence          = data.get('valence')
    tempo            = data.get('tempo')
    time_signature   = data.get('time_signature')

    # Predecir popularidad
    popularity = predict_song_popularity(
        track_genre=track_genre,
        duration_ms=duration_ms,
        explicit=explicit,
        danceability=danceability,
        energy=energy,
        key=key,
        loudness=loudness,
        mode=mode,
        speechiness=speechiness,
        acousticness=acousticness,
        instrumentalness=instrumentalness,
        liveness=liveness,
        valence=valence,
        tempo=tempo,
        time_signature=time_signature
    )

    # Guardar en base de datos
    new_song = Song()
    new_song.track_genre      = track_genre
    new_song.duration_ms      = duration_ms
    new_song.explicit         = explicit
    new_song.danceability     = danceability
    new_song.energy           = energy
    new_song.key              = key
    new_song.loudness         = loudness
    new_song.mode             = mode
    new_song.speechiness      = speechiness
    new_song.acousticness     = acousticness
    new_song.instrumentalness = instrumentalness
    new_song.liveness         = liveness
    new_song.valence          = valence
    new_song.tempo            = tempo
    new_song.time_signature   = time_signature
    new_song.popularity       = popularity

    db.session.add(new_song)
    db.session.commit()

    schema = SongSchema()
    return jsonify(schema.dump(new_song)), 201


# GET /song — devuelve todas las canciones registradas
@app.route('/song', methods=['GET'])
def get_songs():
    songs = Song.query.all()
    schema = SongSchema(many=True)
    return jsonify(schema.dump(songs)), 200


# GET /song/<id> — devuelve una canción por ID
@app.route('/song/<int:id>', methods=['GET'])
def get_song_by_id(id):
    song = Song.query.get(id)
    if not song:
        return jsonify({'message': 'Canción no encontrada'}), 404
    schema = SongSchema()
    return jsonify(schema.dump(song)), 200


# PUT /song/<id> — actualiza los datos y recalcula la predicción
@app.route('/song/<int:id>', methods=['PUT'])
def update_song(id):
    song = Song.query.get(id)
    if not song:
        return jsonify({'message': 'Canción no encontrada'}), 404

    data = request.json

    track_genre      = data.get('track_genre',      song.track_genre)
    duration_ms      = data.get('duration_ms',      song.duration_ms)
    explicit         = data.get('explicit',          song.explicit)
    danceability     = data.get('danceability',      song.danceability)
    energy           = data.get('energy',            song.energy)
    key              = data.get('key',               song.key)
    loudness         = data.get('loudness',          song.loudness)
    mode             = data.get('mode',              song.mode)
    speechiness      = data.get('speechiness',       song.speechiness)
    acousticness     = data.get('acousticness',      song.acousticness)
    instrumentalness = data.get('instrumentalness',  song.instrumentalness)
    liveness         = data.get('liveness',          song.liveness)
    valence          = data.get('valence',           song.valence)
    tempo            = data.get('tempo',             song.tempo)
    time_signature   = data.get('time_signature',    song.time_signature)

    popularity = predict_song_popularity(
        track_genre=track_genre, duration_ms=duration_ms, explicit=explicit,
        danceability=danceability, energy=energy, key=key, loudness=loudness,
        mode=mode, speechiness=speechiness, acousticness=acousticness,
        instrumentalness=instrumentalness, liveness=liveness, valence=valence,
        tempo=tempo, time_signature=time_signature
    )

    song.track_genre      = track_genre
    song.duration_ms      = duration_ms
    song.explicit         = explicit
    song.danceability     = danceability
    song.energy           = energy
    song.key              = key
    song.loudness         = loudness
    song.mode             = mode
    song.speechiness      = speechiness
    song.acousticness     = acousticness
    song.instrumentalness = instrumentalness
    song.liveness         = liveness
    song.valence          = valence
    song.tempo            = tempo
    song.time_signature   = time_signature
    song.popularity       = popularity

    db.session.commit()

    schema = SongSchema()
    return jsonify(schema.dump(song)), 200


# DELETE /song/<id> — elimina una canción
@app.route('/song/<int:id>', methods=['DELETE'])
def delete_song(id):
    song = Song.query.get(id)
    if not song:
        return jsonify({'message': 'Canción no encontrada'}), 404

    db.session.delete(song)
    db.session.commit()
    return jsonify({'message': 'Canción eliminada correctamente'}), 200


if __name__ == '__main__':
    app.run(debug=True)
