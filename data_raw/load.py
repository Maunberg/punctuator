import requests
import gzip

def d_f(url, save_path):
  response = requests.get(url, stream=True, verify=False)
  if response.status_code == 200:
    with open(save_path, 'wb') as f:
      for chunk in response.iter_content(1024):
        if chunk:
          f.write(chunk)
    print(f"Файл успешно скачан: {save_path}")
    with gzip.open(save_path, 'rt', encoding='utf-8', errors='ignore') as file:
        lines = file.read()
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(lines)
  else:
    print(f"Ошибка при скачивании файла: {response.status_code}")



#d_f('https://object.pouta.csc.fi/OPUS-Books/v1/mono/ru.txt.gz', 'books_ru.txt')

#d_f('https://object.pouta.csc.fi/OPUS-Books/v1/mono/en.txt.gz', 'books_en.txt')

#d_f('https://object.pouta.csc.fi/OPUS-NeuLab-TedTalks/v1/mono/ru.txt.gz', 'tedtalks_ru.txt')

#d_f('https://object.pouta.csc.fi/OPUS-NeuLab-TedTalks/v1/mono/en.txt.gz', 'tedtalks_en.txt')

d_f('https://object.pouta.csc.fi/OPUS-ELRC-EMEA/v1/mono/de.txt.gz', 'elrc_de.txt')

d_f('https://object.pouta.csc.fi/OPUS-Tanzil/v1/mono/de.txt.gz', 'tanzil_de.txt')

d_f('https://object.pouta.csc.fi/OPUS-wikimedia/v20230407/mono/fr.txt.gz', 'wikim_fr.txt')

d_f('https://object.pouta.csc.fi/OPUS-EUbookshop/v2/mono/es.txt.gz', 'eubook_es.txt')

d_f('https://object.pouta.csc.fi/OPUS-EUbookshop/v2/mono/pt.txt.gz', 'eubook_pt.txt')

#d_f('')
