from flask import Flask
from ai4bharat.transliteration import XlitEngine

app = Flask(__name__)

sinhala_translit = XlitEngine(lang2use=["si"], rescore=False, beam_width=10, src_script_type = "en")
tamil_translit = XlitEngine(lang2use=["ta"], rescore=False, beam_width=10, src_script_type = "en")

@app.route('/si/<string:input_text>')
def sinhala(input_text):
    sinhala_text = sinhala_translit.translit_sentence(input_text)

    return sinhala_text

@app.route('/ta/<string:input_text>')
def tamil(input_text):
    tamil_text = tamil_translit.translit_sentence(input_text)

    return tamil_text