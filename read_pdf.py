from pypdf import PdfReader

reader = PdfReader("ML_Portfolio_Project (1).pdf")
text = ""
# Read first 3 pages
for i in range(min(3, len(reader.pages))):
    text += reader.pages[i].extract_text() + "\n"

print(text)
