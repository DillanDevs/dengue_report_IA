from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import traceback

app = Flask(__name__)
CORS(app)  # Habilitar CORS para la aplicación Flask

try:
    # Cargar el modelo
    clf = joblib.load('model.pkl')
except FileNotFoundError:
    print("Error: No se encontró el archivo 'model.pkl'. Asegúrate de que el archivo esté en el directorio correcto.")
except Exception as e:
    print("Error al cargar el modelo desde 'model.pkl':")
    print(traceback.format_exc())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Extraer los datos de la solicitud JSON
    tavg = data['tavg']
    tmin = data['tmin']
    tmax = data['tmax']
    prcp = data['prcp']
    wdir = data['wdir']
    wspd = data['wspd']
    pres = data['pres']

    new_data = [[tavg, tmin, tmax, prcp, wdir, wspd, pres]]

    try:
        # Realizar la predicción
        prediction = clf.predict(new_data)

        # Devolver el resultado de la predicción
        result = 'Es probable que se reporten casos de dengue bajo estas condiciones climáticas.' if prediction[0] == 1 else 'Es poco probable que se reporten casos de dengue bajo estas condiciones climáticas.'
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
