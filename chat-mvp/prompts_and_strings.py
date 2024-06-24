SI_SYSTEM_PROMPT = "ඔබ ශ්‍රී ලංකාවේ වී වගාව පිළිබඳ විශේෂඥයෙක්. ගොවියාගේ ප්‍රශ්නයට පිළිතුරු දීමට සපයා ඇති තොරතුරු භාවිතා කරන්න. කෙටි හා සරල පිළිතුරු දෙන්න - අවශ්‍ය නම් පාරිභෝගිකයාගෙන් තවත් ප්‍රශ්න අසන්න. සිංහලෙන් පමණක් පිළිතුරු දෙන්න."
EN_SYSTEM_PROMPT = "You are a paddy cultivation expert in Sri Lanka. Use the information provided to answer the farmer's question. Answer short and simple - ask the customer more questions if necessary. Answer only in English."
TA_SYSTEM_PROMPT = "நீங்கள் இலங்கையில் நெல் சாகுபடி நிபுணர். விவசாயியின் கேள்விக்கு பதிலளிக்க, வழங்கப்பட்ட தகவலைப் பயன்படுத்தவும். சுருக்கமாகவும் எளிமையாகவும் பதிலளிக்கவும் - தேவைப்பட்டால் வாடிக்கையாளரிடம் மேலும் கேள்விகளைக் கேளுங்கள். தமிழில் மட்டும் பதில் சொல்லுங்கள்." 

def get_system_prompt(lang):
  if lang == "en":
    return EN_SYSTEM_PROMPT 
  elif lang == "ta":
    return TA_SYSTEM_PROMPT
  else:
    return SI_SYSTEM_PROMPT
  
SI_TITLE = "ගොවි-මිතු​රු AI 👨🏾‍🌾"
EN_TITLE = "Govi-Mithuru AI 👨🏾‍🌾"
TA_TITLE = "உழவர் தோழன் AI 👨🏾‍🌾"

def get_title(lang):
  if lang == "en":
    return EN_TITLE
  elif lang == "ta":
    return TA_TITLE
  else:
    return SI_TITLE
  
SI_INFO = "ශ්‍රී ලංකාවේ වී වගා කිරීම පිළිබඳව ඔබට ඇති ඕනෑම ප්‍රශ්නයක් අසන්න"
EN_INFO = "Ask any question related to paddy cultivation in Sri Lanka"
TA_INFO = "இலங்கையில் நெல் சாகுபடி தொடர்பான ஏதேனும் கேள்விகளைக் கேளுங்கள்"
  
def get_info(lang):
  if lang == "en":
    return EN_INFO
  elif lang == "ta":
    return TA_INFO
  else:
    return SI_INFO
  

def format_message_with_context(lang, user_input, context):
  if lang== "si":
    return f"ප්රශ්නය: {user_input}\n\nඅදාල තොරතුරු: '{context}'"
  elif lang == "ta":
    return f"கேள்வி: {user_input}\n\nதொடர்புடைய தகவல்: '{context}'"
  else:
    return f"Question: {user_input}\n\nRelevant Information: '{context}'"
