import pandas as pd
import ast
import re
import json
from fractions import Fraction


INPUT_CSV = "RecipeNLG_dataset.csv"
OUTPUT_JSON = "parsed_recipes.json"
NROWS = 10000

UNIT_MAP = {
    "c": "cup",
    "c.": "cup",
    "cup": "cup",
    "cups": "cup",
    "tbsp": "tbsp",
    "tbsp.": "tbsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "tsp": "tsp",
    "tsp.": "tsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "oz": "oz",
    "oz.": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "lb": "lb",
    "lb.": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "ml": "ml",
    "l": "l",
    "pkg": "package",
    "pkg.": "package",
    "package": "package",
    "packages": "package",
    "can": "can",
    "cans": "can",
    "jar": "jar",
    "jars": "jar",
    "carton": "carton",
    "cartons": "carton",
    "box": "box",
    "boxes": "box",
    "bottle": "bottle",
    "bottles": "bottle"
}


def parse_quantity(text):
    text = text.strip()

    mixed_match = re.match(r"^(\d+)\s+(\d+/\d+)", text)
    if mixed_match:
        whole = int(mixed_match.group(1))
        frac = float(Fraction(mixed_match.group(2)))
        return whole + frac, mixed_match.group(0)

    frac_match = re.match(r"^(\d+/\d+)", text)
    if frac_match:
        return float(Fraction(frac_match.group(1))), frac_match.group(1)

    decimal_match = re.match(r"^(\d+(?:\.\d+)?)", text)
    if decimal_match:
        return float(decimal_match.group(1)), decimal_match.group(1)

    return None, None


def normalize_unit(token):
    token = token.lower().strip(",.()")
    return UNIT_MAP.get(token)


def parse_ingredient(raw):
    original = raw.strip()
    working = original

    quantity = None
    unit = None
    package_size = None

    qty, qty_text = parse_quantity(working)
    if qty is not None:
        quantity = qty
        working = working[len(qty_text):].strip()

    # capture parenthetical package size, e.g. (16 oz.)
    if working.startswith("("):
        paren_match = re.match(r"^\(([^)]*)\)\s*(.*)", working)
        if paren_match:
            package_size = paren_match.group(1).strip()
            working = paren_match.group(2).strip()

    tokens = working.split()

    if tokens:
        maybe_unit = normalize_unit(tokens[0])
        if maybe_unit:
            unit = maybe_unit
            working = " ".join(tokens[1:]).strip()

    ingredient_text = working.strip(" ,")

    return {
        "raw": original,
        "quantity": quantity,
        "unit": unit,
        "package_size": package_size,
        "ingredient_text": ingredient_text
    }


def parse_list_column(text):
    try:
        value = ast.literal_eval(text)
        if isinstance(value, list):
            return value
        return []
    except Exception:
        return []


def build_aligned_ingredients(raw_ingredients_str, ner_str):
    raw_ingredients = parse_list_column(raw_ingredients_str)
    ner_ingredients = parse_list_column(ner_str)

    parsed = [parse_ingredient(item) for item in raw_ingredients if isinstance(item, str)]

    # align by index
    for i, item in enumerate(parsed):
        if i < len(ner_ingredients) and isinstance(ner_ingredients[i], str):
            item["ner_ingredient"] = ner_ingredients[i].strip()
        else:
            item["ner_ingredient"] = None

    return parsed


def main():
    df = pd.read_csv(INPUT_CSV, nrows=NROWS)

    print("Original shape:", df.shape)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.dropna(subset=["title"])
    df = df[df["source"] == "Gathered"]

    print("After filtering:", df.shape)

    df["ingredients_parsed"] = df.apply(
        lambda row: build_aligned_ingredients(row["ingredients"], row["NER"]),
        axis=1
    )

    parsed_recipes = df.apply(
        lambda row: {
            "title": row["title"],
            "ingredients_parsed": row["ingredients_parsed"],
            "directions": row["directions"],
            "link": row["link"]
        },
        axis=1
    ).tolist()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed_recipes, f, ensure_ascii=False, indent=2)

    print(f"Saved {OUTPUT_JSON}")
    print("Total parsed recipes:", len(parsed_recipes))
    print("Example parsed recipe:")
    print(json.dumps(parsed_recipes[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()