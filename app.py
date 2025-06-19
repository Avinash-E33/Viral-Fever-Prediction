from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# 1. Load CSVs
# ─────────────────────────────────────────────────────────────
dengu_df   = pd.read_csv(r'C:\Users\dkath\Desktop\project\Dengue-Dataset.csv')
typhoid_df = pd.read_csv(r'C:\Users\dkath\Desktop\project\typhoid 1.csv')

# ─────────────────────────────────────────────────────────────
# 2. Rename “long” columns to short, code-friendly names
#    (Only needs to be done once, right after loading)
# ─────────────────────────────────────────────────────────────
dengue_rename = {
    'Hemoglobin(g/dl)'            : 'Hemoglobin',
    'Neutrophils(%)'              : 'Neutrophils',
    'Lymphocytes(%)'              : 'Lymphocytes',
    'Monocytes(%)'                : 'Monocytes',
    'Eosinophils(%)'              : 'Eosinophils',
    'HCT(%)'                      : 'HCT',
    'MCV(fl)'                     : 'MCV',
    'MCH(pg)'                     : 'MCH',
    'MCHC(g/dl)'                  : 'MCHC',
    'RDW-CV(%)'                   : 'RDW_CV',
    'Total Platelet Count(/cumm)' : 'Platelets',
    'MPV(fl)'                     : 'MPV',
    'PDW(%)'                      : 'PDW',
    'PCT(%)'                      : 'PCT',
    'Total WBC count(/cumm)'      : 'WBC'
}
dengu_df.rename(columns=dengue_rename, inplace=True)

typhoid_rename = {
    'Hemoglobin (g/dL)'      : 'Hemoglobin',
    'Platelet Count'         : 'Platelet_Count',
    'Blood Culture Bacteria' : 'Blood_Culture',
    'Urine Culture Bacteria' : 'Urine_Culture',
    'Calcium (mg/dL)'        : 'Calcium',
    'Potassium (mmol/L)'     : 'Potassium'
}
typhoid_df.rename(columns=typhoid_rename, inplace=True)

# ─────────────────────────────────────────────────────────────
# 3. Label-encode categorical columns & train models
# ─────────────────────────────────────────────────────────────
le_d_gen = LabelEncoder().fit(dengu_df['Gender'])
le_d_res = LabelEncoder().fit(dengu_df['Result'])

dengu_df['Gender'] = le_d_gen.transform(dengu_df['Gender'])
dengu_df['Result'] = le_d_res.transform(dengu_df['Result'])

X_d = dengu_df.drop('Result', axis=1)
y_d = dengu_df['Result']

model_d = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_d.fit(X_d, y_d)

le_t = {}
for col in ['Gender', 'Blood_Culture', 'Urine_Culture', 'Severity']:
    le = LabelEncoder().fit(typhoid_df[col])
    typhoid_df[col] = le.transform(typhoid_df[col])
    le_t[col] = le

X_t = typhoid_df.drop('Severity', axis=1)
y_t = typhoid_df['Severity']

model_t = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model_t.fit(X_t, y_t)

# ─────────────────────────────────────────────────────────────
# 4. Flask routes
# ─────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 4-A. Gather form input (short names, same as HTML)
    user_data = {
        'Gender'        : request.form.get('gender'),
        'Age'           : int(request.form.get('age', 0)),
        'Hemoglobin'    : float(request.form.get('hemoglobin', 0.0)),
        'Neutrophils'   : int(request.form.get('neutrophils', 0)),
        'Lymphocytes'   : int(request.form.get('lymphocytes', 0)),
        'Monocytes'     : int(request.form.get('monocytes', 0)),
        'Eosinophils'   : int(request.form.get('eosinophils', 0)),
        'RBC'           : float(request.form.get('rbc', 0.0)),
        'HCT'           : float(request.form.get('hct', 0.0)),
        'MCV'           : float(request.form.get('mcv', 0.0)),
        'MCH'           : float(request.form.get('mch', 0.0)),
        'MCHC'          : float(request.form.get('mchc', 0.0)),
        'RDW_CV'        : float(request.form.get('rdw_cv', 0.0)),
        'Platelets'     : int(request.form.get('platelets', 0)),
        'MPV'           : float(request.form.get('mpv', 0.0)),
        'PDW'           : float(request.form.get('pdw', 0.0)),
        'PCT'           : float(request.form.get('pct', 0.0)),
        'WBC'           : float(request.form.get('wbc', 0.0)),
        'Platelet_Count': int(request.form.get('platelets', 0)),  # for Typhoid
        'Blood_Culture' : request.form.get('blood_culture'),
        'Urine_Culture' : request.form.get('urine_culture'),
        'Calcium'       : float(request.form.get('calcium', 0.0)),
        'Potassium'     : float(request.form.get('potassium', 0.0))
    }

    # 4-B. Dengue prediction
    df_d_input = pd.DataFrame([user_data])
    df_d_input['Gender'] = le_d_gen.transform(df_d_input['Gender'])
    df_d_input = df_d_input[X_d.columns]           # guarantee correct order/cols
    pred_d     = model_d.predict(df_d_input)[0]
    result_dengue = le_d_res.inverse_transform([pred_d])[0].upper()

    # 4-C. Typhoid only if Dengue is NEGATIVE
    if result_dengue == 'NEGATIVE':
        df_t_input = pd.DataFrame([{
            'Gender'               : user_data['Gender'],
            'Age'                  : user_data['Age'],
            'Hemoglobin'           : user_data['Hemoglobin'],
            'Platelet_Count'       : user_data['Platelet_Count'],
            'Blood_Culture'        : user_data['Blood_Culture'],
            'Urine_Culture'        : user_data['Urine_Culture'],
            'Calcium'              : user_data['Calcium'],
            'Potassium'            : user_data['Potassium']
        }])

        # encode categoricals
        for col in ['Gender', 'Blood_Culture', 'Urine_Culture']:
            df_t_input[col] = le_t[col].transform(df_t_input[col])

        pred_t = model_t.predict(df_t_input)[0]
        sev    = le_t['Severity'].inverse_transform([pred_t])[0]
        sev    = {'moderate': 'Medium'}.get(sev.lower(), sev).upper()

        return render_template(
            'index.html',
            result_dengue=result_dengue,
            result_typhoid=sev
        )

    return render_template('index.html', result_dengue=result_dengue)


if __name__ == '__main__':
    app.run(debug=True)
