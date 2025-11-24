from pypdf import PdfReader

reader = PdfReader("ML_Portfolio_Project (1).pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(text)
