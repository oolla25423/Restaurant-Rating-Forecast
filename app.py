"""
Flask web-приложение: Предсказание рейтинга ресторана (Вариант 14)
Датасет: enhanced_zomato_dataset_clean.csv (Zomato, Индия)
"""
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_dir, 'model.pkl'), 'rb') as f:
    bundle = pickle.load(f)

model       = bundle['model']
scaler      = bundle['scaler']
le_cuisine  = bundle['le_cui']
le_city     = bundle['le_loc']
le_rest_type = bundle['le_rt']
features    = bundle['features']
cuisines    = bundle['cuisines']
cities      = bundle['locations']
rest_types  = bundle['rest_types']

HTML = """
<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Restaurant Rating Predictor</title>
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',sans-serif;background:#f0f2f5;color:#2c3e50}
    .header{background:linear-gradient(135deg,#e74c3c,#c0392b);color:white;padding:22px 32px}
    .header h1{font-size:1.5rem;font-weight:700}
    .header p{font-size:.88rem;opacity:.85;margin-top:4px}
    .container{max-width:780px;margin:28px auto;padding:0 16px}
    .card{background:white;border-radius:12px;padding:26px;
          box-shadow:0 2px 12px rgba(0,0,0,.08);margin-bottom:18px}
    .card h2{font-size:1rem;color:#e74c3c;margin-bottom:16px;
             padding-bottom:8px;border-bottom:1px solid #eee}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    .field label{display:block;font-size:.78rem;font-weight:600;color:#7f8c8d;
                 margin-bottom:5px;text-transform:uppercase;letter-spacing:.4px}
    select,input[type=number]{width:100%;padding:9px 11px;border:1.5px solid #dde1e7;
      border-radius:7px;font-size:.93rem;background:#fafbfc;transition:border-color .2s}
    select:focus,input:focus{outline:none;border-color:#e74c3c;background:white}
    .radio-group{display:flex;gap:14px}
    .radio-group label{display:flex;align-items:center;gap:5px;font-size:.93rem;
                       cursor:pointer;color:#2c3e50;font-weight:500;
                       text-transform:none;letter-spacing:0}
    .radio-group input[type=radio]{width:auto}
    button{width:100%;padding:12px;background:#e74c3c;color:white;border:none;
           border-radius:8px;font-size:.97rem;font-weight:600;cursor:pointer;
           margin-top:6px;transition:background .2s}
    button:hover{background:#c0392b}
    #result{display:none}
    .res-high  {border-left:5px solid #2ecc71;background:#f0faf4}
    .res-low   {border-left:5px solid #e74c3c;background:#fef5f5}
    .res-label {font-size:1.4rem;font-weight:700;margin-bottom:6px}
    .prob-row  {font-size:.88rem;color:#555;margin-top:10px}
    .prob-bar  {background:#eee;border-radius:4px;height:8px;margin:4px 0;overflow:hidden}
    .prob-fill {height:100%;border-radius:4px;transition:width .5s ease}
    .tip{background:#f8f9fa;border-radius:8px;padding:11px 13px;
         font-size:.86rem;color:#555;margin-top:10px;line-height:1.55}
    .tip strong{color:#2c3e50}
  </style>
</head>
<body>
<div class="header">
  <h1>Предсказание рейтинга ресторана</h1>
  <p>Zomato Dataset · 826 ресторанов · 17 городов Индии · Gradient Boosting</p>
</div>
<div class="container">
  <div class="card">
    <h2>Параметры ресторана</h2>
    <div class="grid">
      <div class="field">
        <label>Кухня</label>
        <select id="cuisine">
          {% for c in cuisines %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
        </select>
      </div>
      <div class="field">
        <label>Город</label>
        <select id="city">
          {% for c in cities %}<option value="{{ c }}">{{ c }}</option>{% endfor %}
        </select>
      </div>
      <div class="field">
        <label>Тип ресторана</label>
        <select id="rest_type">
          {% for r in rest_types %}<option value="{{ r }}">{{ r }}</option>{% endfor %}
        </select>
      </div>
      <div class="field">
        <label>Средняя цена блюда (₹)</label>
        <input type="number" id="approx_cost" value="250" min="10" max="5000">
      </div>
      <div class="field">
        <label>Всего голосов</label>
        <input type="number" id="votes" value="500" min="0" max="50000">
      </div>
      <div class="field">
        <label>Онлайн-заказ</label>
        <div class="radio-group">
          <label><input type="radio" name="online_order" value="1"> Да</label>
          <label><input type="radio" name="online_order" value="0" checked> Нет</label>
        </div>
      </div>
      <div class="field">
        <label>Бронирование столика</label>
        <div class="radio-group">
          <label><input type="radio" name="book_table" value="1"> Да</label>
          <label><input type="radio" name="book_table" value="0" checked> Нет</label>
        </div>
      </div>
    </div>
    <button onclick="predict()">Предсказать рейтинг</button>
  </div>

  <div class="card" id="result">
    <h2>Результат</h2>
    <div class="res-label" id="res-label"></div>
    <div class="prob-row">
      Вероятность высокого рейтинга (≥ 4.0): <strong id="p-high">—</strong>
      <div class="prob-bar"><div class="prob-fill" id="bar-high"
           style="background:#2ecc71;width:0%"></div></div>
      Вероятность низкого / среднего (< 4.0): <strong id="p-low">—</strong>
      <div class="prob-bar"><div class="prob-fill" id="bar-low"
           style="background:#e74c3c;width:0%"></div></div>
    </div>
    <div class="tip" id="tip"></div>
  </div>
</div>
<script>
async function predict(){
  const data={
    cuisine:       document.getElementById('cuisine').value,
    city:          document.getElementById('city').value,
    rest_type:     document.getElementById('rest_type').value,
    online_order:  +document.querySelector('input[name=online_order]:checked').value,
    book_table:    +document.querySelector('input[name=book_table]:checked').value,
    votes:         +document.getElementById('votes').value,
    approx_cost:   +document.getElementById('approx_cost').value,
  };
  const r=await(await fetch('/predict',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})).json();
  const el=document.getElementById('result');
  el.style.display='block';
  el.className='card '+(r.label==='High'?'res-high':'res-low');
  document.getElementById('res-label').textContent=
    r.label==='High'?'Высокий рейтинг (≥ 4.0)':'Низкий / Средний рейтинг (< 4.0)';
  document.getElementById('p-high').textContent=(r.prob_high*100).toFixed(1)+'%';
  document.getElementById('p-low').textContent =(r.prob_low*100).toFixed(1)+'%';
  document.getElementById('bar-high').style.width=(r.prob_high*100)+'%';
  document.getElementById('bar-low').style.width =(r.prob_low*100)+'%';
  document.getElementById('tip').innerHTML=r.tip;
}
</script>
</body>
</html>
"""

TIPS = {
    'High': (
        "<strong>Прогноз: высокий рейтинг.</strong> "
        "Ресторан с такими параметрами вероятно пользуется высокой оценкой клиентов. "
        "Ключевые факторы: разнообразное меню с активными бестселлерами, "
        "хорошее соотношение цены к средним по кухне и городу."
    ),
    'Low': (
        "<strong>Прогноз: низкий / средний рейтинг.</strong> "
        "Рекомендации для улучшения:<br>"
        "• Увеличить число позиций меню и выделить бестселлеры<br>"
        "• Собирать голоса и отзывы — активность клиентов повышает рейтинг<br>"
        "• Пересмотреть ценовое позиционирование относительно конкурентов<br>"
        "• Сфокусироваться на популярных в городе кухнях"
    ),
}


@app.route('/')
def index():
    return render_template_string(HTML, cuisines=cuisines, cities=cities, rest_types=rest_types)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        online_order_enc = int(data['online_order'])
        book_table_enc = int(data['book_table'])
        votes = float(data['votes'])
        approx_cost = float(data['approx_cost'])
        
        if data['cuisine'] not in cuisines:
            return jsonify({'error': f'Кухня "{data["cuisine"]}" не найдена'}), 400
        if data['city'] not in cities:
            return jsonify({'error': f'Город "{data["city"]}" не найден'}), 400
        if data['rest_type'] not in rest_types:
            return jsonify({'error': f'Тип ресторана "{data["rest_type"]}" не найден'}), 400
            
        cuisine_enc = int(le_cuisine.transform([data['cuisine']])[0])
        city_enc = int(le_city.transform([data['city']])[0])
        rest_type_enc = int(le_rest_type.transform([data['rest_type']])[0])
    except (KeyError, ValueError, IndexError) as e:
        return jsonify({'error': f'Ошибка в данных: {str(e)}'}), 400

    # Создание row с правильными признаками
    row = {
        'online_order_enc': online_order_enc,
        'book_table_enc': book_table_enc,
        'votes': votes,
        'approx_cost': approx_cost,
        'location_enc': city_enc,
        'rest_type_enc': rest_type_enc,
        'cuisines_enc': cuisine_enc,
    }

    # Создание DataFrame и масштабирование
    X = pd.DataFrame([row], columns=features)
    X.fillna(0, inplace=True)
    X_scaled = scaler.transform(X.values)
    
    # Предсказание
    pred = int(model.predict(X_scaled)[0])
    proba = model.predict_proba(X_scaled)[0]

    label = 'High' if pred == 1 else 'Low'
    return jsonify({
        'label': label,
        'prob_high': round(float(proba[1]), 4),
        'prob_low': round(float(proba[0]), 4),
        'tip': TIPS[label],
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API — JSON in, JSON out."""
    return predict()


if __name__ == '__main__':
    app.run(debug=True, port=5000)
