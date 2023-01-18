import os
import arxiv
from pypdf import PdfReader

def parse_paper(path):
  print("Parsing paper")
  reader = PdfReader(path)
  number_of_pages = len(reader.pages)
  print(f"Total number of pages: {number_of_pages}")
  paper_text = []
  for i in range(number_of_pages):
    page = reader.pages[i]
    page_text = []

    def visitor_body(text, cm, tm, fontDict, fontSize):
      x = tm[4]
      y = tm[5]
      # ignore header/footer
      if (y > 50 and y < 720) and (len(text.strip()) > 1):
        page_text.append({
          'fontsize': fontSize,
          'text': text.strip().replace('\x03', ''),
          'x': x,
          'y': y
        })

    _ = page.extract_text(visitor_text=visitor_body)

    blob_font_size = None
    blob_text = ''
    processed_text = []

    for t in page_text:
      if t['fontsize'] == blob_font_size:
        blob_text += f" {t['text']}"
      else:
        if blob_font_size is not None and len(blob_text) > 1:
          processed_text.append({
            'fontsize': blob_font_size,
            'text': blob_text,
            'page': i
          })
        blob_font_size = t['fontsize']
        blob_text = t['text']
    paper_text += processed_text
  return paper_text

def check_folders():
    paths = {
        'data_path' : 'data',
        'images_path' : 'data/pdf'
    }
    
    # Check whether the specified path exists or not
    notExist = list(({file_type: path for (file_type, path) in paths.items() if not os.path.exists(path)}).values())
    
    if notExist:
        print(f'Folder {notExist} does not exist. We will created')
        # Create a new directory because it does not exist
        for folder in notExist:
            os.makedirs(folder)
            print(f"The new directory {folder} is created!")
  
# download link from arxiv. Ex https://arxiv.org/abs/2301.05586
        
def download_arxiv(url):
    id_from_url = url.split("/")[-1:]
    paper = next(arxiv.Search(id_list=id_from_url).results())
    # Download the PDF to a specified directory with a custom filename.
    paper.download_pdf(dirpath='data/pdf/', filename="downloaded-paper.pdf")
    