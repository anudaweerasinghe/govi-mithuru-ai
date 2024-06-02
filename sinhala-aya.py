from transformers import AutoModelForSeq2SeqLM, AutoTokenizer 

input = "ඔබට සිංහල කතා කළ හැකිද?"

checkpoint = "CohereForAI/aya-101"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
aya_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, load_in_8bit=True)

tokenized_input = tokenizer.encode(input, return_tensors="pt")
tokenized_output = aya_model.generate(tokenized_input, max_length=256)

print(tokenizer.decode(tokenized_output[0]))