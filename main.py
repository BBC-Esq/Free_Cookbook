import json
import math
import sys
import sqlite3
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from thefuzz import fuzz

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QAbstractItemView,
    QHeaderView,
)

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas


DB_FILENAME = "cookbook.db"


@dataclass(frozen=True)
class Recipe:
    id: int
    name: str
    category: str
    gluten_free: bool
    notes: str
    instructions: str
    prep_time: str
    cook_time: str
    total_time: str
    servings: str


@dataclass(frozen=True)
class Ingredient:
    id: int
    recipe_id: int
    name: str
    quantity: str
    unit: str
    preparation: str
    optional: bool
    sort_order: int


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def db_path() -> Path:
    return app_dir() / DB_FILENAME


def open_db(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    return con


def db_has_schema(con: sqlite3.Connection) -> bool:
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('recipes','ingredients')"
    )
    found = {r["name"] for r in cur.fetchall()}
    return "recipes" in found and "ingredients" in found


def create_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS recipes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            gluten_free INTEGER NOT NULL DEFAULT 0,
            notes TEXT NOT NULL DEFAULT '',
            instructions TEXT NOT NULL DEFAULT '',
            prep_time TEXT NOT NULL DEFAULT '',
            cook_time TEXT NOT NULL DEFAULT '',
            total_time TEXT NOT NULL DEFAULT '',
            servings TEXT NOT NULL DEFAULT ''
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ingredients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recipe_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            quantity TEXT NOT NULL DEFAULT '',
            unit TEXT NOT NULL DEFAULT '',
            preparation TEXT NOT NULL DEFAULT '',
            optional INTEGER NOT NULL DEFAULT 0,
            sort_order INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(recipe_id) REFERENCES recipes(id) ON DELETE CASCADE
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_recipes_name ON recipes(name)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ingredients_recipe ON ingredients(recipe_id)")


def insert_seed_data(con: sqlite3.Connection, recipes: list) -> None:
    for r in recipes:
        name = str(r.get("name", "")).strip()
        if not name:
            continue
        category = str(r.get("category", "")).strip() or "Uncategorized"
        gluten_free = 1 if bool(r.get("gluten_free", False)) else 0
        notes = str(r.get("notes", "") or "")
        instructions = str(r.get("instructions", "") or "")
        prep_time = str(r.get("prep_time", "") or "")
        cook_time = str(r.get("cook_time", "") or "")
        total_time = str(r.get("total_time", "") or "")
        servings = str(r.get("servings", "") or "")
        cur = con.execute(
            "INSERT INTO recipes (name, category, gluten_free, notes, instructions, prep_time, cook_time, total_time, servings) VALUES (?,?,?,?,?,?,?,?,?)",
            (name, category, gluten_free, notes, instructions, prep_time, cook_time, total_time, servings),
        )
        recipe_id = int(cur.lastrowid)
        for idx, ing in enumerate(r.get("ingredients", []) or []):
            iname = str(ing.get("name", "")).strip()
            if not iname:
                continue
            quantity = str(ing.get("quantity", "") or "")
            unit = str(ing.get("unit", "") or "")
            preparation = str(ing.get("preparation", "") or "")
            optional = 1 if bool(ing.get("optional", False)) else 0
            sort_order = int(ing.get("sort_order", idx))
            con.execute(
                "INSERT INTO ingredients (recipe_id, name, quantity, unit, preparation, optional, sort_order) VALUES (?,?,?,?,?,?,?)",
                (recipe_id, iname, quantity, unit, preparation, optional, sort_order),
            )


def init_db(path: Path) -> sqlite3.Connection:
    from seed_data import SEED_RECIPES
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    create_schema(con)
    insert_seed_data(con, SEED_RECIPES)
    con.commit()
    return con


def fetch_categories(con: sqlite3.Connection) -> List[str]:
    cur = con.execute("SELECT DISTINCT category FROM recipes ORDER BY category COLLATE NOCASE")
    cats = [r["category"] or "" for r in cur.fetchall()]
    cats = [c for c in cats if c.strip()]
    return cats


def fetch_recipes(con: sqlite3.Connection) -> List[Recipe]:
    cur = con.execute(
        "SELECT id, name, category, gluten_free, notes, instructions, prep_time, cook_time, total_time, servings FROM recipes ORDER BY name COLLATE NOCASE"
    )
    out: List[Recipe] = []
    for r in cur.fetchall():
        out.append(
            Recipe(
                id=int(r["id"]),
                name=str(r["name"] or ""),
                category=str(r["category"] or ""),
                gluten_free=bool(r["gluten_free"] or 0),
                notes=str(r["notes"] or ""),
                instructions=str(r["instructions"] or ""),
                prep_time=str(r["prep_time"] or ""),
                cook_time=str(r["cook_time"] or ""),
                total_time=str(r["total_time"] or ""),
                servings=str(r["servings"] or ""),
            )
        )
    return out


def fetch_ingredients_for_recipe(con: sqlite3.Connection, recipe_id: int) -> List[Ingredient]:
    cur = con.execute(
        """
        SELECT id, recipe_id, name, quantity, unit, preparation, optional, sort_order
        FROM ingredients
        WHERE recipe_id = ?
        ORDER BY sort_order ASC, name COLLATE NOCASE ASC
        """,
        (recipe_id,),
    )
    out: List[Ingredient] = []
    for r in cur.fetchall():
        out.append(
            Ingredient(
                id=int(r["id"]),
                recipe_id=int(r["recipe_id"]),
                name=str(r["name"] or ""),
                quantity=str(r["quantity"] or ""),
                unit=str(r["unit"] or ""),
                preparation=str(r["preparation"] or ""),
                optional=bool(r["optional"] or 0),
                sort_order=int(r["sort_order"] or 0),
            )
        )
    return out


def fetch_ingredient_names_by_recipe(con: sqlite3.Connection) -> Dict[int, List[str]]:
    cur = con.execute("SELECT recipe_id, name FROM ingredients ORDER BY recipe_id, sort_order")
    out: Dict[int, List[str]] = {}
    for r in cur.fetchall():
        rid = int(r["recipe_id"])
        if rid not in out:
            out[rid] = []
        out[rid].append(str(r["name"] or ""))
    return out


VALID_CATEGORIES = {"Appetizer", "Beverage", "Dessert", "Main", "Salad", "Sauce", "Side", "Soup"}

VALID_RECIPE_KEYS = {"name", "category", "gluten_free", "notes", "instructions",
                     "prep_time", "cook_time", "total_time", "servings", "ingredients"}

REQUIRED_RECIPE_KEYS = {"name", "category", "gluten_free", "instructions", "ingredients"}

VALID_INGREDIENT_KEYS = {"name", "quantity", "unit", "preparation", "optional", "sort_order"}

REQUIRED_INGREDIENT_KEYS = {"name", "quantity", "unit", "sort_order"}


def validate_llm_response(raw_text: str) -> Tuple[Optional[Dict], List[str]]:
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        lines = raw_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    errors: List[str] = []

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, [f"Response is not valid JSON: {exc}"]

    if not isinstance(data, dict):
        return None, ["Response must be a JSON object (dict), not " + type(data).__name__]

    extra_keys = set(data.keys()) - VALID_RECIPE_KEYS
    if extra_keys:
        errors.append(f"Unexpected top-level keys: {sorted(extra_keys)}. Remove them.")

    for key in REQUIRED_RECIPE_KEYS:
        if key not in data:
            errors.append(f"Missing required key: \"{key}\"")

    if errors:
        return None, errors

    if not isinstance(data["name"], str) or not data["name"].strip():
        errors.append("\"name\" must be a non-empty string.")

    if data["category"] not in VALID_CATEGORIES:
        errors.append(
            f"\"category\" must be one of {sorted(VALID_CATEGORIES)}. Got: \"{data['category']}\"")

    if not isinstance(data["gluten_free"], bool):
        errors.append("\"gluten_free\" must be a boolean (true/false), not " + type(data["gluten_free"]).__name__)

    if not isinstance(data["instructions"], str) or not data["instructions"].strip():
        errors.append("\"instructions\" must be a non-empty string.")

    for opt_key in ("notes", "prep_time", "cook_time", "total_time", "servings"):
        if opt_key in data:
            if data[opt_key] is None:
                errors.append(f"\"{opt_key}\" must be a string, not null. Use \"\" for empty.")
            elif not isinstance(data[opt_key], str):
                errors.append(f"\"{opt_key}\" must be a string, got {type(data[opt_key]).__name__}")

    if not isinstance(data["ingredients"], list):
        errors.append("\"ingredients\" must be an array.")
        return None, errors

    if len(data["ingredients"]) == 0:
        errors.append("\"ingredients\" must contain at least one ingredient.")

    seen_sort_orders = set()
    for idx, ing in enumerate(data["ingredients"]):
        prefix = f"ingredients[{idx}]"

        if not isinstance(ing, dict):
            errors.append(f"{prefix}: must be a JSON object, not {type(ing).__name__}")
            continue

        ing_extra = set(ing.keys()) - VALID_INGREDIENT_KEYS
        if ing_extra:
            errors.append(f"{prefix}: unexpected keys: {sorted(ing_extra)}. Remove them.")

        for req in REQUIRED_INGREDIENT_KEYS:
            if req not in ing:
                errors.append(f"{prefix}: missing required key \"{req}\"")

        if "name" in ing:
            if not isinstance(ing["name"], str) or not ing["name"].strip():
                errors.append(f"{prefix}: \"name\" must be a non-empty string.")

        if "quantity" in ing:
            if not isinstance(ing["quantity"], str):
                errors.append(
                    f"{prefix}: \"quantity\" must be a string, got {type(ing['quantity']).__name__}."
                    " Use \"\" for unmeasured, fractions like \"1/2\", or whole numbers like \"3\".")

        if "unit" in ing:
            if not isinstance(ing["unit"], str):
                errors.append(f"{prefix}: \"unit\" must be a string, got {type(ing['unit']).__name__}")

        if "sort_order" in ing:
            if not isinstance(ing["sort_order"], int):
                errors.append(f"{prefix}: \"sort_order\" must be an integer, got {type(ing['sort_order']).__name__}")
            else:
                if ing["sort_order"] in seen_sort_orders:
                    errors.append(f"{prefix}: duplicate sort_order {ing['sort_order']}")
                seen_sort_orders.add(ing["sort_order"])

        if "preparation" in ing:
            if ing["preparation"] is None:
                errors.append(f"{prefix}: \"preparation\" must be a string, not null. Use \"\".")
            elif not isinstance(ing["preparation"], str):
                errors.append(f"{prefix}: \"preparation\" must be a string")

        if "optional" in ing:
            if not isinstance(ing["optional"], bool):
                errors.append(f"{prefix}: \"optional\" must be a boolean (true/false)")

    if errors:
        return None, errors

    return data, []


def build_retry_prompt(original_response: str, validation_errors: List[str]) -> str:
    error_block = "\n".join(f"  - {e}" for e in validation_errors)
    return (
        "Your previous response failed structural validation. You MUST fix every issue listed below "
        "and return a corrected JSON object. Do not add explanatory text. Output ONLY the raw JSON "
        "object starting with { and ending with }.\n"
        "\n"
        "VALIDATION ERRORS:\n"
        f"{error_block}\n"
        "\n"
        "YOUR PREVIOUS RESPONSE (which had errors):\n"
        f"{original_response}\n"
        "\n"
        "RULES REMINDER:\n"
        "- Output must be a single JSON object parseable by json.loads()\n"
        "- Required recipe keys: name (string), category (one of: Appetizer, Beverage, Dessert, "
        "Main, Salad, Sauce, Side, Soup), gluten_free (boolean), instructions (string), "
        "ingredients (array)\n"
        "- Optional recipe keys (use \"\" if empty, never null): notes, prep_time, cook_time, "
        "total_time, servings\n"
        "- Each ingredient object requires: name (string), quantity (string), unit (string), "
        "sort_order (integer)\n"
        "- Optional ingredient keys: preparation (string, default \"\"), optional (boolean, default false)\n"
        "- No extra keys anywhere. No null values. No markdown fences.\n"
        "- quantity must be a string: \"1/2\", \"3\", \"\", etc. Never a number type.\n"
        "\n"
        "Return the corrected JSON now."
    )


def insert_recipe_from_llm(con: sqlite3.Connection, data: Dict) -> int:
    name = data["name"].strip()
    category = data["category"]
    gluten_free = 1 if data["gluten_free"] else 0
    notes = (data.get("notes") or "").strip()
    instructions = data["instructions"].strip()
    prep_time = (data.get("prep_time") or "").strip()
    cook_time = (data.get("cook_time") or "").strip()
    total_time = (data.get("total_time") or "").strip()
    servings = (data.get("servings") or "").strip()

    cur = con.execute(
        "INSERT INTO recipes (name, category, gluten_free, notes, instructions, "
        "prep_time, cook_time, total_time, servings) VALUES (?,?,?,?,?,?,?,?,?)",
        (name, category, gluten_free, notes, instructions,
         prep_time, cook_time, total_time, servings),
    )
    recipe_id = int(cur.lastrowid)

    for ing in data["ingredients"]:
        iname = ing["name"].strip()
        quantity = (ing.get("quantity") or "").strip()
        unit = (ing.get("unit") or "").strip()
        preparation = (ing.get("preparation") or "").strip()
        optional = 1 if ing.get("optional", False) else 0
        sort_order = int(ing.get("sort_order", 0))
        con.execute(
            "INSERT INTO ingredients (recipe_id, name, quantity, unit, preparation, optional, sort_order) "
            "VALUES (?,?,?,?,?,?,?)",
            (recipe_id, iname, quantity, unit, preparation, optional, sort_order),
        )

    con.commit()
    return recipe_id


def load_prompt_template() -> str:
    prompt_path = app_dir() / "sample_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def build_extraction_prompt(recipe_text: str) -> str:
    template = load_prompt_template()
    return template.replace("{recipe_text}", recipe_text)


MAX_LLM_RETRIES = 2


def process_llm_recipe(con: sqlite3.Connection, send_fn, recipe_text: str) -> Tuple[bool, str, int]:
    prompt = build_extraction_prompt(recipe_text)
    response = send_fn(prompt)
    data, errors = validate_llm_response(response)

    for attempt in range(MAX_LLM_RETRIES):
        if data is not None:
            break
        retry_prompt = build_retry_prompt(response, errors)
        response = send_fn(retry_prompt)
        data, errors = validate_llm_response(response)

    if data is None:
        return False, "Validation failed after retries:\n" + "\n".join(errors), 0

    recipe_id = insert_recipe_from_llm(con, data)
    return True, data["name"], recipe_id


def normalize_key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


_frac_re = re.compile(r"(\d+)\s*/\s*(\d+)")
_mixed_re = re.compile(r"^(\d+)\s+(\d+)\s*/\s*(\d+)$")
_whole_re = re.compile(r"^(\d+(?:\.\d+)?)$")
_range_re = re.compile(r"^(\S+)\s*[-–]\s*(\S+)$")
_compound_re = re.compile(r"(.+?)\s*\+\s*(.+)")


def parse_quantity(text: str) -> Optional[Fraction]:
    text = (text or "").strip()
    if not text:
        return None
    m = _mixed_re.match(text)
    if m:
        w, n, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if d == 0:
            return None
        return Fraction(w) + Fraction(n, d)
    m = _frac_re.match(text)
    if m:
        n, d = int(m.group(1)), int(m.group(2))
        if d == 0:
            return None
        return Fraction(n, d)
    m = _whole_re.match(text)
    if m:
        try:
            return Fraction(m.group(1)).limit_denominator(1000)
        except (ValueError, ZeroDivisionError):
            return None
    m = _range_re.match(text)
    if m:
        lo = parse_quantity(m.group(1))
        hi = parse_quantity(m.group(2))
        if lo is not None and hi is not None:
            return (lo + hi) / 2
        return lo or hi
    return None


def fraction_to_str(f: Fraction) -> str:
    if f <= 0:
        return "0"
    whole = int(f)
    remainder = f - whole
    if remainder == 0:
        return str(whole)
    remainder = remainder.limit_denominator(16)
    if remainder == 0:
        return str(whole)
    if whole == 0:
        return f"{remainder.numerator}/{remainder.denominator}"
    return f"{whole} {remainder.numerator}/{remainder.denominator}"


UNIT_ALIASES: Dict[str, str] = {
    "cup": "cup", "cups": "cup", "c": "cup", "c.": "cup",
    "tbsp": "tbsp", "tablespoon": "tbsp", "tablespoons": "tbsp",
    "tbs": "tbsp", "tbs.": "tbsp", "tbsp.": "tbsp", "t": "tbsp",
    "tsp": "tsp", "teaspoon": "tsp", "teaspoons": "tsp",
    "tsp.": "tsp",
    "oz": "oz", "ounce": "oz", "ounces": "oz", "oz.": "oz",
    "lb": "lb", "lbs": "lb", "pound": "lb", "pounds": "lb",
    "lb.": "lb", "lbs.": "lb",
    "gal": "gal", "gallon": "gal", "gallons": "gal",
    "qt": "qt", "quart": "qt", "quarts": "qt",
    "pt": "pt", "pint": "pt", "pints": "pt",
    "fl oz": "fl_oz", "fluid ounce": "fl_oz", "fluid ounces": "fl_oz",
    "ml": "ml", "milliliter": "ml", "milliliters": "ml", "millilitre": "ml",
    "l": "liter", "liter": "liter", "liters": "liter", "litre": "liter",
    "g": "g", "gram": "g", "grams": "g", "gramme": "g",
    "kg": "kg", "kilogram": "kg", "kilograms": "kg",
    "dash": "dash", "pinch": "pinch",
    "clove": "clove", "cloves": "clove",
    "head": "head", "heads": "head",
    "sprig": "sprig", "sprigs": "sprig",
    "stick": "stick", "sticks": "stick",
    "stalk": "stalk", "stalks": "stalk",
    "slice": "slice", "slices": "slice",
    "large": "large", "medium": "medium", "small": "small",
    "can": "can", "cans": "can",
    "package": "package", "packages": "package", "pkg": "package",
    "bunch": "bunch", "bunches": "bunch",
    "ear": "ear", "ears": "ear",
}

VOLUME_TO_TSP: Dict[str, Fraction] = {
    "tsp": Fraction(1),
    "tbsp": Fraction(3),
    "fl_oz": Fraction(6),
    "cup": Fraction(48),
    "pt": Fraction(96),
    "qt": Fraction(192),
    "gal": Fraction(768),
    "ml": Fraction(48, 237),
    "liter": Fraction(48000, 237),
}

WEIGHT_TO_OZ: Dict[str, Fraction] = {
    "oz": Fraction(1),
    "lb": Fraction(16),
    "g": Fraction(1000, 28349),
    "kg": Fraction(1000000, 28349),
}

def canonical_unit(raw: str) -> str:
    raw = (raw or "").strip().lower()
    canned_re = re.match(r"^(\d+(?:\.\d+)?)\s*oz\s+(can|cans)$", raw)
    if canned_re:
        return raw
    return UNIT_ALIASES.get(raw, raw)


def convert_to_base(qty: Fraction, unit: str) -> Optional[Tuple[Fraction, str]]:
    if unit in VOLUME_TO_TSP:
        return (qty * VOLUME_TO_TSP[unit], "vol_tsp")
    if unit in WEIGHT_TO_OZ:
        return (qty * WEIGHT_TO_OZ[unit], "wt_oz")
    return None


SHOPPING_VOLUME: List[Tuple[str, Fraction, str]] = [
    ("cup", Fraction(48), "cup"),
    ("tbsp", Fraction(3), "tbsp"),
    ("tsp", Fraction(1), "tsp"),
]

SHOPPING_WEIGHT: List[Tuple[str, Fraction, str]] = [
    ("lb", Fraction(16), "lb"),
    ("oz", Fraction(1), "oz"),
]


UNIT_THRESHOLDS: Dict[str, Fraction] = {
    "cup": Fraction(1, 4),
    "lb": Fraction(1, 2),
}


def best_display_unit(base_val: Fraction, measure_type: str) -> Tuple[str, Fraction]:
    if measure_type == "vol_tsp":
        table = SHOPPING_VOLUME
    elif measure_type == "wt_oz":
        table = SHOPPING_WEIGHT
    else:
        return ("", base_val)
    for unit_name, factor, display in table:
        converted = base_val / factor
        threshold = UNIT_THRESHOLDS.get(display, Fraction(1))
        if converted >= threshold:
            return (display, converted)
    return (table[-1][2], base_val / table[-1][1])


ROUND_FRACTIONS = [
    Fraction(1, 8), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2),
    Fraction(2, 3), Fraction(3, 4), Fraction(1),
]


def round_to_common(f: Fraction) -> Fraction:
    if f <= 0:
        return Fraction(0)
    whole = int(f)
    remainder = f - whole
    if remainder == 0:
        return f
    candidates = [Fraction(whole) + rf for rf in ROUND_FRACTIONS]
    at_or_above = [c for c in candidates if c >= f]
    if at_or_above:
        return min(at_or_above)
    return Fraction(whole + 1)


MAX_SEARCH_RESULTS = 20

_tokenize_re = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _tokenize_re.findall(text.lower())


class SearchIndex:
    def __init__(self, recipes: List["Recipe"], ingredient_map: Dict[int, List[str]]):
        self.recipes = recipes
        self.ingredient_map = ingredient_map
        self.name_tokens: Dict[int, List[str]] = {}
        self.ing_tokens: Dict[int, List[str]] = {}
        self.name_lower: Dict[int, str] = {}
        self.ing_text_lower: Dict[int, str] = {}
        self.doc_freqs: Counter = Counter()
        self.doc_count = len(recipes)
        self.tfidf_vecs: Dict[int, Dict[str, float]] = {}

        for r in recipes:
            ntoks = tokenize(r.name)
            self.name_tokens[r.id] = ntoks
            self.name_lower[r.id] = r.name.lower()
            ing_names = ingredient_map.get(r.id, [])
            itoks = []
            for iname in ing_names:
                itoks.extend(tokenize(iname))
            self.ing_tokens[r.id] = itoks
            self.ing_text_lower[r.id] = " ".join(n.lower() for n in ing_names)
            all_toks = set(ntoks + itoks)
            for t in all_toks:
                self.doc_freqs[t] += 1

        for r in recipes:
            all_toks = self.name_tokens[r.id] + self.ing_tokens[r.id]
            tf = Counter(all_toks)
            total = len(all_toks) or 1
            vec: Dict[str, float] = {}
            for tok, count in tf.items():
                df = self.doc_freqs.get(tok, 1)
                idf = math.log((self.doc_count + 1) / (df + 1)) + 1
                vec[tok] = (count / total) * idf
            self.tfidf_vecs[r.id] = vec

    def _cosine(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        keys = set(vec_a) & set(vec_b)
        if not keys:
            return 0.0
        dot = sum(vec_a[k] * vec_b[k] for k in keys)
        mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
        mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def search(self, query: str) -> List[Tuple[int, float]]:
        query = query.strip()
        if not query:
            return [(r.id, 0.0) for r in self.recipes]

        ql = query.lower()
        qtoks = tokenize(query)
        if not qtoks:
            return [(r.id, 0.0) for r in self.recipes]

        qtf = Counter(qtoks)
        total_q = len(qtoks) or 1
        qvec: Dict[str, float] = {}
        for tok, count in qtf.items():
            df = self.doc_freqs.get(tok, 1)
            idf = math.log((self.doc_count + 1) / (df + 1)) + 1
            qvec[tok] = (count / total_q) * idf

        scored: List[Tuple[int, float]] = []
        for r in self.recipes:
            score = 0.0

            name_l = self.name_lower[r.id]
            if ql in name_l:
                score += 100.0
                if name_l == ql:
                    score += 50.0
                elif name_l.startswith(ql):
                    score += 25.0

            ing_l = self.ing_text_lower[r.id]
            if ql in ing_l:
                score += 40.0

            tfidf_score = self._cosine(qvec, self.tfidf_vecs[r.id])
            score += tfidf_score * 30.0

            if len(ql) >= 4:
                name_fuzz = fuzz.partial_ratio(ql, name_l)
                if name_fuzz >= 75:
                    score += (name_fuzz / 100.0) * 15.0

                if ing_l:
                    ing_fuzz = fuzz.partial_ratio(ql, ing_l)
                    if ing_fuzz >= 85:
                        score += (ing_fuzz / 100.0) * 10.0

            scored.append((r.id, score))

        scored.sort(key=lambda x: (-x[1], self.name_lower.get(x[0], "")))
        return scored


def export_pdf_lines(path: Path, title: str, lines: List[str]) -> None:
    c = Canvas(str(path), pagesize=letter)
    width, height = letter
    x = 72
    y = height - 72
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 28
    c.setFont("Helvetica", 11)
    line_height = 14
    for line in lines:
        if y < 72:
            c.showPage()
            y = height - 72
            c.setFont("Helvetica", 11)
        c.drawString(x, y, line)
        y -= line_height
    c.save()


def wrap_text(text: str, font_name: str, font_size: int, max_width: float) -> List[str]:
    from reportlab.pdfbase.pdfmetrics import stringWidth
    wrapped: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            wrapped.append("")
            continue
        words = paragraph.split()
        if not words:
            wrapped.append("")
            continue
        current = words[0]
        for w in words[1:]:
            test = current + " " + w
            if stringWidth(test, font_name, font_size) <= max_width:
                current = test
            else:
                wrapped.append(current)
                current = w
        wrapped.append(current)
    return wrapped


def export_recipe_card_pdf(path: Path, recipe: "Recipe", ingredients: List["Ingredient"]) -> None:
    c = Canvas(str(path), pagesize=letter)
    width, height = letter
    margin = 72
    max_w = width - 2 * margin
    y = height - margin

    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, recipe.name)
    y -= 24

    meta_parts = [f"Category: {recipe.category}"]
    if recipe.prep_time:
        meta_parts.append(f"Prep: {recipe.prep_time}")
    if recipe.cook_time:
        meta_parts.append(f"Cook: {recipe.cook_time}")
    if recipe.total_time:
        meta_parts.append(f"Total: {recipe.total_time}")
    if recipe.servings:
        meta_parts.append(f"Servings: {recipe.servings}")
    if recipe.gluten_free:
        meta_parts.append("Gluten-free")
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, "  |  ".join(meta_parts))
    y -= 10
    c.setLineWidth(0.5)
    c.line(margin, y, width - margin, y)
    y -= 18

    def check_page(needed: float) -> float:
        nonlocal y
        if y - needed < margin:
            c.showPage()
            y = height - margin
        return y

    if ingredients:
        check_page(20)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, "Ingredients")
        y -= 18
        c.setFont("Helvetica", 11)
        for ing in ingredients:
            check_page(14)
            parts = []
            if (ing.quantity or "").strip():
                parts.append(ing.quantity.strip())
            if (ing.unit or "").strip():
                parts.append(ing.unit.strip())
            parts.append(ing.name.strip())
            if (ing.preparation or "").strip():
                parts.append(f"({ing.preparation.strip()})")
            if ing.optional:
                parts.append("[optional]")
            c.drawString(margin + 12, y, "\u2022  " + " ".join(parts))
            y -= 14

    if (recipe.instructions or "").strip():
        y -= 8
        check_page(20)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, "Instructions")
        y -= 18
        c.setFont("Helvetica", 11)
        wrapped = wrap_text(recipe.instructions.strip(), "Helvetica", 11, max_w - 12)
        for line in wrapped:
            check_page(14)
            c.drawString(margin + 12, y, line)
            y -= 14

    if (recipe.notes or "").strip():
        y -= 8
        check_page(20)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(margin, y, "Notes")
        y -= 18
        c.setFont("Helvetica", 11)
        wrapped = wrap_text(recipe.notes.strip(), "Helvetica", 11, max_w - 12)
        for line in wrapped:
            check_page(14)
            c.drawString(margin + 12, y, line)
            y -= 14

    c.save()


class MainWindow(QMainWindow):
    def __init__(self, con: sqlite3.Connection):
        super().__init__()
        self.con = con
        self.setWindowTitle("Cookbook")
        self.resize(1100, 780)

        self.all_recipes: List[Recipe] = []
        self.recipe_by_id: Dict[int, Recipe] = {}
        self.selected_recipes: List[Tuple[int, int]] = []

        self.toolbar = QToolBar("Main")
        self.addToolBar(self.toolbar)

        self.action_refresh = QAction("Refresh", self)
        self.action_refresh.triggered.connect(self.refresh_data)
        self.toolbar.addAction(self.action_refresh)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search recipes or ingredients...")
        self._search_timer = QTimer()
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(750)
        self._search_timer.timeout.connect(self.apply_filters)
        self.search.textChanged.connect(self._on_search_changed)
        left_layout.addWidget(self.search)

        self.category = QComboBox()
        self.category.currentIndexChanged.connect(self.apply_filters)
        left_layout.addWidget(self.category)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_recipe_context_menu)
        left_layout.addWidget(self.list_widget, 1)

        button_row = QHBoxLayout()
        self.btn_export_card = QPushButton("Export Recipe Card PDF")
        self.btn_export_card.clicked.connect(self.export_recipe_card)
        self.btn_export_card.setEnabled(False)
        button_row.addWidget(self.btn_export_card)

        self.btn_export_recipe = QPushButton("Export Ingredients PDF")
        self.btn_export_recipe.clicked.connect(self.export_selected_recipe_pdf)
        self.btn_export_recipe.setEnabled(False)
        button_row.addWidget(self.btn_export_recipe)
        left_layout.addLayout(button_row)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)

        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs, 1)

        self.tab_details = QWidget()
        details_layout = QVBoxLayout(self.tab_details)

        self.lbl_details = QLabel("")
        self.lbl_details.setTextFormat(Qt.PlainText)
        self.lbl_details.setWordWrap(True)
        details_layout.addWidget(self.lbl_details)

        self.ingredients_table = QTableWidget(0, 5)
        self.ingredients_table.setHorizontalHeaderLabels(["Ingredient", "Quantity", "Unit", "Prep", "Optional"])
        self.ingredients_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.ingredients_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.ingredients_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.ingredients_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.ingredients_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.ingredients_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.ingredients_table.setSelectionMode(QAbstractItemView.NoSelection)
        details_layout.addWidget(self.ingredients_table, 1)

        self.instructions = QTextEdit()
        self.instructions.setReadOnly(True)
        self.instructions.setPlaceholderText("Instructions")
        details_layout.addWidget(self.instructions, 1)

        self.tabs.addTab(self.tab_details, "Recipe Details")

        self.tab_shopping = QWidget()
        shopping_layout = QVBoxLayout(self.tab_shopping)

        self.shopping_table = QTableWidget(0, 5)
        self.shopping_table.setHorizontalHeaderLabels(["Ingredient", "Quantity", "Unit", "Prep/Notes", "Sources"])
        self.shopping_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.shopping_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.shopping_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.shopping_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.shopping_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.shopping_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.shopping_table.setSelectionMode(QAbstractItemView.NoSelection)
        shopping_layout.addWidget(self.shopping_table, 1)

        self.tabs.addTab(self.tab_shopping, "Shopping List")

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root_layout.addWidget(sep)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 4, 8, 8)

        bottom_header = QHBoxLayout()
        bottom_header.addWidget(QLabel("Selected Recipes"))
        bottom_header.addStretch()

        self.btn_clear_selected = QPushButton("Clear All")
        self.btn_clear_selected.clicked.connect(self._clear_selected_recipes)
        bottom_header.addWidget(self.btn_clear_selected)

        self.btn_export_shopping = QPushButton("Export Shopping List PDF")
        self.btn_export_shopping.clicked.connect(self.export_shopping_list_pdf)
        self.btn_export_shopping.setEnabled(False)
        bottom_header.addWidget(self.btn_export_shopping)

        bottom_layout.addLayout(bottom_header)

        self.selected_scroll = QScrollArea()
        self.selected_scroll.setWidgetResizable(True)
        self.selected_scroll.setFixedHeight(170)
        self.selected_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.selected_container = QWidget()
        self.selected_layout = QVBoxLayout(self.selected_container)
        self.selected_layout.setContentsMargins(4, 4, 4, 4)
        self.selected_layout.setSpacing(4)
        self.selected_layout.addStretch()
        self.selected_scroll.setWidget(self.selected_container)
        bottom_layout.addWidget(self.selected_scroll)

        root_layout.addWidget(bottom)

        self.refresh_data()

    def refresh_data(self) -> None:
        self.all_recipes = fetch_recipes(self.con)
        self.recipe_by_id = {r.id: r for r in self.all_recipes}
        self.ingredient_names = fetch_ingredient_names_by_recipe(self.con)
        self.search_index = SearchIndex(self.all_recipes, self.ingredient_names)

        cats = fetch_categories(self.con)
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItem("All Categories")
        for c in cats:
            self.category.addItem(c)
        self.category.blockSignals(False)

        self.apply_filters()
        self.on_selection_changed()

    def _on_search_changed(self) -> None:
        self._search_timer.start()

    def apply_filters(self) -> None:
        text = (self.search.text() or "").strip()
        cat = self.category.currentText()
        use_cat = cat != "All Categories"

        scored = self.search_index.search(text)
        has_query = bool(text)

        if use_cat:
            scored = [(rid, s) for rid, s in scored if (self.recipe_by_id.get(rid) and self.recipe_by_id[rid].category == cat)]

        if has_query:
            scored = [(rid, s) for rid, s in scored if s > 0]
            scored = scored[:MAX_SEARCH_RESULTS]

        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for rid, _ in scored:
            r = self.recipe_by_id.get(rid)
            if not r:
                continue
            item = QListWidgetItem(r.name)
            item.setData(Qt.UserRole, r.id)
            item.setData(Qt.UserRole + 1, r.category)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def _show_recipe_context_menu(self, pos) -> None:
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        rid = item.data(Qt.UserRole)
        if rid is None:
            return
        rid = int(rid)
        menu = QMenu(self)
        already = any(sr[0] == rid for sr in self.selected_recipes)
        if already:
            act = menu.addAction("Already in selected recipes")
            act.setEnabled(False)
        else:
            act_add = menu.addAction("Add to Selected Recipes")
            act_add.triggered.connect(lambda: self._add_to_selected(rid))
        menu.exec(self.list_widget.viewport().mapToGlobal(pos))

    def _add_to_selected(self, rid: int) -> None:
        if any(sr[0] == rid for sr in self.selected_recipes):
            return
        self.selected_recipes.append((rid, 1))
        self._rebuild_selected_panel()
        self._update_shopping_from_selected()

    def _remove_from_selected(self, rid: int) -> None:
        self.selected_recipes = [(r, m) for r, m in self.selected_recipes if r != rid]
        self._rebuild_selected_panel()
        self._update_shopping_from_selected()

    def _set_multiplier(self, rid: int, val: int) -> None:
        self.selected_recipes = [(r, val if r == rid else m) for r, m in self.selected_recipes]
        self._update_shopping_from_selected()

    def _clear_selected_recipes(self) -> None:
        self.selected_recipes.clear()
        self._rebuild_selected_panel()
        self._update_shopping_from_selected()

    def _rebuild_selected_panel(self) -> None:
        while self.selected_layout.count() > 0:
            child = self.selected_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        for rid, mult in self.selected_recipes:
            r = self.recipe_by_id.get(rid)
            if not r:
                continue
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(8)

            lbl = QLabel(r.name)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            row_layout.addWidget(lbl)

            row_layout.addWidget(QLabel("Servings \u00d7"))

            spin = QSpinBox()
            spin.setMinimum(1)
            spin.setMaximum(20)
            spin.setValue(mult)
            spin.setFixedWidth(55)
            captured_rid = rid
            spin.valueChanged.connect(lambda v, r=captured_rid: self._set_multiplier(r, v))
            row_layout.addWidget(spin)

            btn_rm = QPushButton("Remove")
            btn_rm.setFixedWidth(70)
            btn_rm.clicked.connect(lambda _, r=captured_rid: self._remove_from_selected(r))
            row_layout.addWidget(btn_rm)

            self.selected_layout.addWidget(row_widget)

        self.selected_layout.addStretch()
        self.btn_export_shopping.setEnabled(len(self.selected_recipes) > 0)

    def selected_recipe_ids(self) -> List[int]:
        ids: List[int] = []
        for it in self.list_widget.selectedItems():
            rid = it.data(Qt.UserRole)
            if rid is not None:
                ids.append(int(rid))
        return ids

    def on_selection_changed(self) -> None:
        ids = self.selected_recipe_ids()
        single = len(ids) == 1
        self.btn_export_recipe.setEnabled(single)
        self.btn_export_card.setEnabled(single)
        self.update_details(ids)

    def update_details(self, ids: List[int]) -> None:
        self.ingredients_table.setRowCount(0)
        self.instructions.setPlainText("")
        if len(ids) != 1:
            if len(ids) == 0:
                self.lbl_details.setText("No recipe selected.")
            else:
                self.lbl_details.setText("Multiple recipes selected. Open the Shopping List tab for consolidated ingredients.")
            return

        r = self.recipe_by_id.get(ids[0])
        if not r:
            self.lbl_details.setText("Recipe not found.")
            return

        gf = "Yes" if r.gluten_free else "No"
        detail_parts = [f"Name: {r.name}", f"Category: {r.category}", f"Gluten-free: {gf}"]
        if r.prep_time:
            detail_parts.append(f"Prep Time: {r.prep_time}")
        if r.cook_time:
            detail_parts.append(f"Cook Time: {r.cook_time}")
        if r.total_time:
            detail_parts.append(f"Total Time: {r.total_time}")
        if r.servings:
            detail_parts.append(f"Servings: {r.servings}")
        if r.notes:
            detail_parts.append(f"Notes: {r.notes}")
        self.lbl_details.setText("\n".join(detail_parts))

        ingredients = fetch_ingredients_for_recipe(self.con, r.id)
        self.ingredients_table.setRowCount(len(ingredients))
        for row, ing in enumerate(ingredients):
            self.ingredients_table.setItem(row, 0, QTableWidgetItem(ing.name))
            self.ingredients_table.setItem(row, 1, QTableWidgetItem(ing.quantity))
            self.ingredients_table.setItem(row, 2, QTableWidgetItem(ing.unit))
            self.ingredients_table.setItem(row, 3, QTableWidgetItem(ing.preparation))
            self.ingredients_table.setItem(row, 4, QTableWidgetItem("Yes" if ing.optional else "No"))

        self.instructions.setPlainText(r.instructions or "")

    def _parse_compound_qty(self, qty_text: str, unit_text: str) -> List[Tuple[Fraction, str]]:
        qty_text = (qty_text or "").strip()
        unit_text = (unit_text or "").strip()
        if not qty_text and not unit_text:
            return []
        m = _compound_re.match(qty_text)
        if m:
            left_part = m.group(1).strip()
            right_part = m.group(2).strip()
            results: List[Tuple[Fraction, str]] = []
            for part in [left_part, right_part]:
                tokens = part.split()
                part_unit = ""
                part_qty_str = part
                for i, tok in enumerate(tokens):
                    if canonical_unit(tok) != tok.lower() or tok.lower() in UNIT_ALIASES:
                        part_unit = tok
                        part_qty_str = " ".join(tokens[:i] + tokens[i+1:])
                        break
                pq = parse_quantity(part_qty_str)
                if pq is not None:
                    results.append((pq, canonical_unit(part_unit) if part_unit else canonical_unit(unit_text)))
            if results:
                return results
        pq = parse_quantity(qty_text)
        if pq is not None:
            return [(pq, canonical_unit(unit_text))]
        return []

    def consolidate_ingredients(self, recipe_mults: List[Tuple[int, int]]) -> List[Dict[str, Any]]:
        accum: Dict[str, Dict[str, Any]] = {}
        for rid, mult in recipe_mults:
            recipe = self.recipe_by_id.get(rid)
            if not recipe:
                continue
            ings = fetch_ingredients_for_recipe(self.con, rid)
            for ing in ings:
                name_key = normalize_key(ing.name)
                if name_key not in accum:
                    accum[name_key] = {
                        "name": ing.name.strip(),
                        "amounts": [],
                        "raw_texts": [],
                        "prep_notes": set(),
                        "sources": set(),
                        "optional_all": True,
                    }
                entry = accum[name_key]
                parsed = self._parse_compound_qty(ing.quantity, ing.unit)
                if parsed:
                    for qty_frac, unit_canon in parsed:
                        entry["amounts"].append((qty_frac * mult, unit_canon))
                else:
                    raw = (ing.quantity or "").strip()
                    if raw:
                        display = raw
                        u = (ing.unit or "").strip()
                        if u:
                            display = f"{raw} {u}"
                        if mult > 1:
                            display = f"{mult}\u00d7 {display}"
                        entry["raw_texts"].append(display)
                prep = (ing.preparation or "").strip()
                if prep:
                    entry["prep_notes"].add(prep)
                entry["sources"].add(recipe.name)
                if not ing.optional:
                    entry["optional_all"] = False

        out: List[Dict[str, Any]] = []
        for _, entry in accum.items():
            vol_base = Fraction(0)
            wt_base = Fraction(0)
            count_by_unit: Dict[str, Fraction] = {}
            has_vol = False
            has_wt = False

            for qty, u in entry["amounts"]:
                converted = convert_to_base(qty, u)
                if converted:
                    base_val, base_type = converted
                    if base_type == "vol_tsp":
                        vol_base += base_val
                        has_vol = True
                    else:
                        wt_base += base_val
                        has_wt = True
                else:
                    if u not in count_by_unit:
                        count_by_unit[u] = Fraction(0)
                    count_by_unit[u] += qty

            parts: List[str] = []
            display_unit_parts: List[str] = []

            if has_vol and vol_base > 0:
                disp_unit, disp_val = best_display_unit(vol_base, "vol_tsp")
                rounded = round_to_common(disp_val)
                parts.append(fraction_to_str(rounded))
                display_unit_parts.append(disp_unit)

            if has_wt and wt_base > 0:
                disp_unit, disp_val = best_display_unit(wt_base, "wt_oz")
                rounded = round_to_common(disp_val)
                parts.append(fraction_to_str(rounded))
                display_unit_parts.append(disp_unit)

            for u, total in sorted(count_by_unit.items()):
                rounded = round_to_common(total)
                parts.append(fraction_to_str(rounded))
                display_unit_parts.append(u)

            for rt in entry["raw_texts"]:
                parts.append(rt)

            qty_str = " + ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
            unit_str = " + ".join(display_unit_parts) if len(display_unit_parts) > 1 else (display_unit_parts[0] if display_unit_parts else "")

            if len(parts) <= 1 and len(display_unit_parts) <= 1:
                pass
            elif len(parts) > 1:
                combined_parts = []
                for i, p in enumerate(parts):
                    u = display_unit_parts[i] if i < len(display_unit_parts) else ""
                    combined_parts.append(f"{p} {u}".strip())
                qty_str = " + ".join(combined_parts)
                unit_str = ""

            prep_notes = "; ".join(sorted(entry["prep_notes"]))
            if entry["optional_all"] and entry["sources"]:
                prep_notes = (prep_notes + " | Optional").strip(" |")

            out.append({
                "name": entry["name"],
                "quantity": qty_str,
                "unit": unit_str,
                "prep": prep_notes,
                "sources": ", ".join(sorted(entry["sources"])),
            })

        out.sort(key=lambda d: normalize_key(d["name"]))
        return out

    def _update_shopping_from_selected(self) -> None:
        self.update_shopping(self.selected_recipes)

    def update_shopping(self, recipe_mults: List[Tuple[int, int]]) -> None:
        rows = self.consolidate_ingredients(recipe_mults) if recipe_mults else []
        self.shopping_table.setRowCount(len(rows))
        for row, d in enumerate(rows):
            self.shopping_table.setItem(row, 0, QTableWidgetItem(d["name"]))
            self.shopping_table.setItem(row, 1, QTableWidgetItem(d["quantity"]))
            self.shopping_table.setItem(row, 2, QTableWidgetItem(d["unit"]))
            self.shopping_table.setItem(row, 3, QTableWidgetItem(d["prep"]))
            self.shopping_table.setItem(row, 4, QTableWidgetItem(d["sources"]))

    def export_selected_recipe_pdf(self) -> None:
        ids = self.selected_recipe_ids()
        if len(ids) != 1:
            return
        r = self.recipe_by_id.get(ids[0])
        if not r:
            return
        ings = fetch_ingredients_for_recipe(self.con, r.id)
        lines: List[str] = []
        for ing in ings:
            parts = []
            if (ing.quantity or "").strip():
                parts.append(ing.quantity.strip())
            if (ing.unit or "").strip():
                parts.append(ing.unit.strip())
            parts.append(ing.name.strip())
            if (ing.preparation or "").strip():
                parts.append(f"({ing.preparation.strip()})")
            if ing.optional:
                parts.append("[Optional]")
            lines.append(" ".join([p for p in parts if p]))

        if not lines:
            lines = ["(No ingredients in database for this recipe)"]

        default_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", r.name.strip()).strip("_") or "recipe"
        path_str, _ = QFileDialog.getSaveFileName(self, "Save PDF", f"{default_name}_ingredients.pdf", "PDF Files (*.pdf)")
        if not path_str:
            return
        export_pdf_lines(Path(path_str), f"Ingredients: {r.name}", lines)

    def export_recipe_card(self) -> None:
        ids = self.selected_recipe_ids()
        if len(ids) != 1:
            return
        r = self.recipe_by_id.get(ids[0])
        if not r:
            return
        ings = fetch_ingredients_for_recipe(self.con, r.id)
        default_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", r.name.strip()).strip("_") or "recipe"
        path_str, _ = QFileDialog.getSaveFileName(self, "Save PDF", f"{default_name}_recipe_card.pdf", "PDF Files (*.pdf)")
        if not path_str:
            return
        export_recipe_card_pdf(Path(path_str), r, ings)

    def export_shopping_list_pdf(self) -> None:
        if not self.selected_recipes:
            return
        rows = self.consolidate_ingredients(self.selected_recipes)
        lines: List[str] = []
        if rows:
            for d in rows:
                parts = []
                if (d["quantity"] or "").strip():
                    parts.append(d["quantity"].strip())
                if (d["unit"] or "").strip():
                    parts.append(d["unit"].strip())
                parts.append(d["name"].strip())
                suffix = []
                if (d["prep"] or "").strip():
                    suffix.append(d["prep"].strip())
                if (d["sources"] or "").strip():
                    suffix.append(f"Sources: {d['sources'].strip()}")
                if suffix:
                    parts.append(f" - {' | '.join(suffix)}")
                lines.append(" ".join([p for p in parts if p]))
        else:
            lines = ["(No ingredients found for the selected recipes)"]

        path_str, _ = QFileDialog.getSaveFileName(self, "Save PDF", "shopping_list.pdf", "PDF Files (*.pdf)")
        if not path_str:
            return
        export_pdf_lines(Path(path_str), "Consolidated Shopping List", lines)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    path = db_path()
    if path.exists():
        con = open_db(path)
        if not db_has_schema(con):
            con.close()
            path.unlink()
            con = init_db(path)
    else:
        con = init_db(path)

    try:
        win = MainWindow(con)
        win.show()
        return app.exec()
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
