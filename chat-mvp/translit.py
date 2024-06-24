from siconv import singlish_to_sinhala
import requests

supported_langs = ["si", "ta"]

def translit(target_lang, user_input_en):
  if target_lang not in supported_langs:
    return user_input_en
  
  translit_model_response = requests.get(f"https://sea-lion-app-8mfcr.ondigitalocean.app/{target_lang}/{user_input_en}")

  if translit_model_response.status_code == 200:
    return translit_model_response.json().get(target_lang)
  else:
    if target_lang == "si":
      return singlish_to_sinhala(user_input_en)
    else:
      return user_input_en