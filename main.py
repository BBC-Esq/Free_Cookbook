import sys
import os
import sqlite3
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QSortFilterProxyModel, QStringListModel
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
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


_number_re = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def try_parse_number(s: str) -> Optional[float]:
    m = _number_re.match(s or "")
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def normalize_key(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())


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


class MainWindow(QMainWindow):
    def __init__(self, con: sqlite3.Connection):
        super().__init__()
        self.con = con
        self.setWindowTitle("Cookbook")
        self.resize(1100, 700)

        self.all_recipes: List[Recipe] = []
        self.recipe_by_id: Dict[int, Recipe] = {}

        self.toolbar = QToolBar("Main")
        self.addToolBar(self.toolbar)

        self.action_refresh = QAction("Refresh", self)
        self.action_refresh.triggered.connect(self.refresh_data)
        self.toolbar.addAction(self.action_refresh)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search recipes...")
        self.search.textChanged.connect(self.apply_filters)
        left_layout.addWidget(self.search)

        self.category = QComboBox()
        self.category.currentIndexChanged.connect(self.apply_filters)
        left_layout.addWidget(self.category)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        left_layout.addWidget(self.list_widget, 1)

        button_row = QHBoxLayout()
        self.btn_export_recipe = QPushButton("Export Selected Recipe Ingredients PDF")
        self.btn_export_recipe.clicked.connect(self.export_selected_recipe_pdf)
        self.btn_export_recipe.setEnabled(False)
        button_row.addWidget(self.btn_export_recipe)

        self.btn_export_shopping = QPushButton("Export Consolidated Shopping List PDF")
        self.btn_export_shopping.clicked.connect(self.export_shopping_list_pdf)
        self.btn_export_shopping.setEnabled(False)
        button_row.addWidget(self.btn_export_shopping)
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

        self.refresh_data()

    def refresh_data(self) -> None:
        self.all_recipes = fetch_recipes(self.con)
        self.recipe_by_id = {r.id: r for r in self.all_recipes}

        cats = fetch_categories(self.con)
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItem("All Categories")
        for c in cats:
            self.category.addItem(c)
        self.category.blockSignals(False)

        self.populate_list()
        self.apply_filters()
        self.on_selection_changed()

    def populate_list(self) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for r in self.all_recipes:
            item = QListWidgetItem(r.name)
            item.setData(Qt.UserRole, r.id)
            item.setData(Qt.UserRole + 1, r.category)
            self.list_widget.addItem(item)
        self.list_widget.blockSignals(False)

    def apply_filters(self) -> None:
        text = (self.search.text() or "").strip().lower()
        cat = self.category.currentText()
        use_cat = cat != "All Categories"

        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            name = (item.text() or "").lower()
            item_cat = item.data(Qt.UserRole + 1) or ""
            visible = True
            if text and text not in name:
                visible = False
            if use_cat and (item_cat or "") != cat:
                visible = False
            item.setHidden(not visible)

    def selected_recipe_ids(self) -> List[int]:
        ids: List[int] = []
        for it in self.list_widget.selectedItems():
            rid = it.data(Qt.UserRole)
            if rid is not None:
                ids.append(int(rid))
        return ids

    def on_selection_changed(self) -> None:
        ids = self.selected_recipe_ids()
        self.btn_export_shopping.setEnabled(len(ids) >= 1)
        self.btn_export_recipe.setEnabled(len(ids) == 1)
        self.update_details(ids)
        self.update_shopping(ids)

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

    def consolidate_ingredients(self, recipe_ids: List[int]) -> List[Dict[str, Any]]:
        accum: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for rid in recipe_ids:
            recipe = self.recipe_by_id.get(rid)
            if not recipe:
                continue
            ings = fetch_ingredients_for_recipe(self.con, rid)
            for ing in ings:
                key = (normalize_key(ing.name), (ing.unit or "").strip().lower())
                if key not in accum:
                    accum[key] = {
                        "name": ing.name.strip(),
                        "unit": (ing.unit or "").strip(),
                        "quantity_value": 0.0,
                        "quantity_texts": [],
                        "quantity_is_numeric": True,
                        "prep_notes": set(),
                        "sources": set(),
                        "optional_any": False,
                    }
                entry = accum[key]
                q = (ing.quantity or "").strip()
                qnum = try_parse_number(q)
                if q == "":
                    entry["quantity_is_numeric"] = False
                elif qnum is None:
                    entry["quantity_is_numeric"] = False
                    entry["quantity_texts"].append(q)
                else:
                    if entry["quantity_is_numeric"]:
                        entry["quantity_value"] += qnum
                    else:
                        entry["quantity_texts"].append(q)
                prep = (ing.preparation or "").strip()
                if prep:
                    entry["prep_notes"].add(prep)
                entry["sources"].add(recipe.name)
                entry["optional_any"] = entry["optional_any"] or bool(ing.optional)

        out: List[Dict[str, Any]] = []
        for _, entry in accum.items():
            if entry["quantity_is_numeric"]:
                qty = entry["quantity_value"]
                qty_str = str(int(qty)) if abs(qty - int(qty)) < 1e-9 else str(round(qty, 3)).rstrip("0").rstrip(".")
            else:
                uniq = []
                for t in entry["quantity_texts"]:
                    if t not in uniq:
                        uniq.append(t)
                qty_str = ", ".join(uniq) if uniq else ""
            prep_notes = "; ".join(sorted(entry["prep_notes"]))
            if entry["optional_any"]:
                prep_notes = (prep_notes + " | Optional").strip(" |")
            out.append(
                {
                    "name": entry["name"],
                    "quantity": qty_str,
                    "unit": entry["unit"],
                    "prep": prep_notes,
                    "sources": ", ".join(sorted(entry["sources"])),
                }
            )

        out.sort(key=lambda d: (normalize_key(d["name"]), (d["unit"] or "").lower()))
        return out

    def update_shopping(self, ids: List[int]) -> None:
        rows = self.consolidate_ingredients(ids) if ids else []
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

    def export_shopping_list_pdf(self) -> None:
        ids = self.selected_recipe_ids()
        if not ids:
            return
        rows = self.consolidate_ingredients(ids)
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
