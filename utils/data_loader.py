import pandas as pd

def load_data(path: str):
    df = pd.read_csv(path)
    # Ожидается столбец: "text", и при наличии — "label"
    return df
