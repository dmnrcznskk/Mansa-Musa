IML
# Zasady kodowania
**Python**
- Dodawanie typów zmiennych i zwracanych przez funkcje. Jeżeli się nie da, to nie piszemy nic, chyba że to argument do funkcji to piszemy `zmienna: any`
- Dodawanie opisu funkcji i klas
- wszystkie moduły dodajemy do folderu `src/nazwa_projektu` lub głębiej
- importy zawsze wyglądają następująco `nazwa_projektu.nazwa_modułu.plik`
- Istnieją 2 tryby odpalenia programu: `poetry run api` odpalające program z aplikacją w FastAPI oraz `poetry run dev`, które służy do wszelkich innych celów. Kolejno funkcje `start_api()` jak i `start_dev()` będą w `main.py` i można zmieniać tylko zawartość tych funkcji. Ewentualne zmiany do uzgodnienia

## Instrukcja instalacji poetry
1. `curl -sSL https://install.python-poetry.org | python3 -`
2. `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc`
3. `source ~/.zshrc`

### Dodawanie zależności
Normalne:
`poetry add nazwa_biblioteki`

Developerskie:
`poetry add nazwa_biblioteki --group dev`

Czasami po zmianach trzeba użyć `poetry install`