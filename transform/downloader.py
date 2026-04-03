from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-zh-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

tokenizer.save_pretrained("./models/opus-mt-zh-en")
model.save_pretrained("./models/opus-mt-zh-en")
