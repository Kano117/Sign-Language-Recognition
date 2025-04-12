import pickle

with open("data\Phoenix-2014T\gloss2ids.pkl", "rb") as f:
    gloss2id = pickle.load(f)
print(gloss2id)