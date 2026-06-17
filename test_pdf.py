from pypdf import PdfReader

reader = PdfReader("documents/cv.pdf")

for page in reader.pages:
    print(page.extract_text())