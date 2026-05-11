import numpy as np
import pickle
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ── Load model & scalers ─────────────────────────────────────────────────────
model             = pickle.load(open('model.pkl',             'rb'))
mx                = pickle.load(open('minmaxscaler.pkl',      'rb'))
sc                = pickle.load(open('standscaler.pkl',       'rb'))
reverse_crop_dict = pickle.load(open('reverse_crop_dict.pkl', 'rb'))

# ── Crop metadata (emoji + short tip shown on result) ────────────────────────
CROP_INFO = {
    'rice':        ('🌾', 'Grows best in flooded paddies with high humidity and rainfall.'),
    'maize':       ('🌽', 'Thrives in warm climates with moderate rainfall and well-drained soil.'),
    'jute':        ('🌿', 'Prefers hot, humid conditions with heavy rainfall.'),
    'cotton':      ('☁️',  'Requires a warm, dry climate with well-drained black soil.'),
    'coconut':     ('🥥', 'Tropical crop needing high humidity and consistent rainfall.'),
    'papaya':      ('🧡', 'Fast-growing; needs warm weather, good drainage, and no waterlogging.'),
    'orange':      ('🍊', 'Citrus crop preferring subtropical climate with mild winters.'),
    'apple':       ('🍎', 'Needs cool winters for dormancy and a mild, dry summer.'),
    'muskmelon':   ('🍈', 'Warm-season fruit requiring sandy loam soil and low humidity.'),
    'watermelon':  ('🍉', 'Loves long, hot summers with sandy well-drained soil.'),
    'grapes':      ('🍇', 'Prefers dry, warm summers and cool winters with well-drained soil.'),
    'mango':       ('🥭', 'Tropical fruit thriving in hot, dry weather during flowering.'),
    'banana':      ('🍌', 'Requires high humidity, heavy rainfall, and rich loamy soil.'),
    'pomegranate': ('🍎', 'Drought-tolerant; prefers semi-arid climate with well-drained soil.'),
    'lentil':      ('🫘', 'Cool-season legume; drought tolerant once established.'),
    'blackgram':   ('🫘', 'Warm, humid conditions; fixes nitrogen in soil.'),
    'mungbean':    ('🌱', 'Short-duration legume suitable for summer and spring seasons.'),
    'mothbeans':   ('🌱', 'Extremely drought-tolerant; ideal for arid and semi-arid zones.'),
    'pigeonpeas':  ('🫘', 'Hardy legume tolerating poor soil and dry conditions.'),
    'kidneybeans': ('🫘', 'Cool-season crop needing fertile, well-drained soil.'),
    'chickpea':    ('🫘', 'Drought-tolerant pulse; prefers cool, dry weather during maturation.'),
    'coffee':      ('☕', 'Shade-loving; thrives in tropical highlands with steady rainfall.'),
}


def recommend(N, P, K, temperature, humidity, ph, rainfall):
    features    = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_features = sc.transform(mx_features)
    pred_num    = model.predict(sc_features)[0]
    crop_name   = reverse_crop_dict.get(pred_num, 'unknown')
    return crop_name


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        N           = float(data['N'])
        P           = float(data['P'])
        K           = float(data['K'])
        temperature = float(data['temperature'])
        humidity    = float(data['humidity'])
        ph          = float(data['ph'])
        rainfall    = float(data['rainfall'])

        crop      = recommend(N, P, K, temperature, humidity, ph, rainfall)
        emoji, tip = CROP_INFO.get(crop, ('🌱', 'A great crop for your conditions.'))

        return jsonify({
            'success': True,
            'crop':    crop.title(),
            'emoji':   emoji,
            'tip':     tip
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
