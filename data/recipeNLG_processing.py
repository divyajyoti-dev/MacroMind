import pandas as pd
import ast
import json

# sample data 10k
df = pd.read_csv("RecipeNLG_dataset.csv", nrows=10000)

print("Original shape:", df.shape)
print("Columns:", list(df.columns))

# drop index column
df = df.drop(columns=["Unnamed: 0"])
# remove missing titles
df = df.dropna(subset=["title"])
# keep only "Gathered" recipes
df = df[df["source"] == "Gathered"]

print("After filtering:", df.shape)

# parse NER to list
def parse_ner(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

df["ingredients_clean"] = df["NER"].apply(parse_ner)

print("example parsed ingredients:", df["ingredients_clean"].iloc[0])
print("parsed ingredient type:", type(df["ingredients_clean"].iloc[0]))

df["num_ingredients"] = df["ingredients_clean"].apply(len)

df = df[df["num_ingredients"] < 30]

print("after removing outliers:", df.shape)

# build structured recipe objects
recipes = df.apply(lambda row: {
    "title": row["title"],
    "ingredients": row["ingredients_clean"],
    "directions": row["directions"],
    "link": row["link"]
}, axis=1).tolist()

print("example recipe:")
print(recipes[0])
print("total cleaned recipes:", len(recipes))

# save cleaned recipes as json
with open("cleaned_recipes.json", "w", encoding="utf-8") as f:
    json.dump(recipes, f, ensure_ascii=False, indent=2)