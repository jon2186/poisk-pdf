import sys
import os
import json
import hashlib
import re
from pathlib import Path

import fitz
import pytesseract
from PIL import Image
from pytesseract import Output

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QListWidget,
    QListWidgetItem, QProgressBar, QMessageBox, QSplitter, QScrollArea
)

APP_NAME = "Поиск по Тобольским епархиальным ведомостям"
CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PDFWordSearch" / "cache"
CACHE_VERSION = "hyphen-phrase-v4"

def resource_path(*parts):
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root.joinpath(*parts)

def configure_bundled_tesseract():
    tess_dir = resource_path("tesseract")
    exe = tess_dir / "tesseract.exe"
    data_dir = tess_dir / "tessdata"
    if exe.exists():
        pytesseract.pytesseract.tesseract_cmd = str(exe)
        os.environ["TESSDATA_PREFIX"] = str(data_dir)
        os.environ["PATH"] = str(tess_dir) + os.pathsep + os.environ.get("PATH", "")

def normalize(s):
    s = s.casefold().replace("ё", "е")
    s = s.translate(str.maketrans({"ѣ": "е", "і": "и", "ѳ": "ф", "ѵ": "и"}))
    # Ignore punctuation for matching while preserving letters and digits.
    return re.sub(r"[^0-9a-zа-я]+", "", s, flags=re.IGNORECASE)

def query_tokens(s):
    # Split the query into separate searchable words.
    return [normalize(part) for part in re.findall(r"[0-9A-Za-zА-Яа-яЁёѢѣІіѲѳѴѵ]+", s) if normalize(part)]

def is_line_hyphenated(text):
    return bool(text) and text.rstrip().endswith(("-", "‐", "‑", "‒", "–", "—", "="))

def strip_line_hyphen(text):
    return text.rstrip().rstrip("-‐‑‒–—=").rstrip()

def make_logical_words(words):
    logical, i = [], 0
    while i < len(words):
        first = words[i]
        if is_line_hyphenated(first["text"]) and i + 1 < len(words):
            second = words[i + 1]
            same_line = (
                first.get("block_num") == second.get("block_num")
                and first.get("par_num") == second.get("par_num")
                and first.get("line_num") == second.get("line_num")
            )
            if not same_line:
                logical.append({
                    "text": strip_line_hyphen(first["text"]) + second["text"].lstrip(),
                    "boxes": [first, second],
                    "conf": min(first.get("conf", 0), second.get("conf", 0))
                })
                i += 2
                continue
        logical.append({"text": first["text"], "boxes": [first], "conf": first.get("conf", 0)})
        i += 1
    return logical

def find_matches(words, query):
    """
    A single token query matches a substring inside an OCR word.
    A multi-token query matches consecutive OCR words in reading order.
    A line-end hyphenated word is already merged by make_logical_words.
    """
    q_tokens = query_tokens(query)
    if not q_tokens:
        return []
    logical = make_logical_words(words)
    tokens = [normalize(w["text"]) for w in logical]
    found = []

    if len(q_tokens) == 1:
        q = q_tokens[0]
        for i, token in enumerate(tokens):
            if q and q in token:
                found.append({
                    "text": logical[i]["text"],
                    "boxes": logical[i]["boxes"],
                    "conf": logical[i]["conf"]
                })
        return found

    n = len(q_tokens)
    for i in range(0, len(tokens) - n + 1):
        if tokens[i:i+n] == q_tokens:
            selected = logical[i:i+n]
            boxes = [box for item in selected for box in item["boxes"]]
            text = " ".join(item["text"] for item in selected)
            confidence = sum(item["conf"] for item in selected) / len(selected)
            found.append({"text": text, "boxes": boxes, "conf": confidence})
    return found

def cache_key(pdf_path):
    p = Path(pdf_path)
    st = p.stat()
    raw = f"{str(p.resolve())}|{st.st_size}|{st.st_mtime_ns}|{CACHE_VERSION}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

class FolderOCRWorker(QThread):
    progress = Signal(int, int, str)
    finished_ok = Signal(object, int, int)
    failed = Signal(str)

    def __init__(self, folder):
        super().__init__()
        self.folder = str(folder)

    def run(self):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            pdf_files = sorted(
                [p for p in Path(self.folder).rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"],
                key=lambda p: str(p).casefold()
            )
            if not pdf_files:
                raise RuntimeError("В выбранной папке не найдено PDF-файлов.")

            records = []
            total_pages = 0
            file_pages = []
            for p in pdf_files:
                try:
                    d = fitz.open(str(p))
                    n = len(d)
                    d.close()
                except Exception:
                    n = 0
                file_pages.append(n)
                total_pages += n
            if total_pages == 0:
                raise RuntimeError("Не удалось открыть PDF-файлы в выбранной папке.")

            done_pages = 0
            skipped_files = []
            for file_index, pdf in enumerate(pdf_files):
                self.progress.emit(done_pages, total_pages, f"Документ {file_index + 1}/{len(pdf_files)}: {pdf.name}")
                cache_file = CACHE_DIR / f"{cache_key(pdf)}.json"
                pages = None
                if cache_file.exists():
                    try:
                        cached = json.loads(cache_file.read_text(encoding="utf-8"))
                        if cached.get("version") == CACHE_VERSION:
                            pages = cached["pages"]
                    except Exception:
                        pages = None

                if pages is None:
                    try:
                        doc = fitz.open(str(pdf))
                        pages = []
                        for page_no, page in enumerate(doc):
                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                            data = pytesseract.image_to_data(
                                image, lang="rus", config="--oem 1 --psm 6",
                                output_type=Output.DICT
                            )
                            words = []
                            for i, raw in enumerate(data["text"]):
                                word = (raw or "").strip()
                                if not word:
                                    continue
                                try:
                                    conf = float(data["conf"][i])
                                except Exception:
                                    conf = -1
                                if conf < 0:
                                    continue
                                words.append({
                                    "text": word,
                                    "left": int(data["left"][i]),
                                    "top": int(data["top"][i]),
                                    "width": int(data["width"][i]),
                                    "height": int(data["height"][i]),
                                    "conf": conf,
                                    "block_num": int(data["block_num"][i]),
                                    "par_num": int(data["par_num"][i]),
                                    "line_num": int(data["line_num"][i]),
                                })
                            pages.append({"page": page_no, "width": pix.width, "height": pix.height, "words": words})
                            done_pages += 1
                            self.progress.emit(done_pages, total_pages, f"{pdf.name} — страница {page_no + 1}/{len(doc)}")
                        doc.close()
                        cache_file.write_text(
                            json.dumps({"version": CACHE_VERSION, "pages": pages}, ensure_ascii=False),
                            encoding="utf-8"
                        )
                    except Exception as e:
                        skipped_files.append(f"{pdf.name}: {e}")
                        done_pages += file_pages[file_index]
                        continue
                else:
                    done_pages += len(pages)
                    self.progress.emit(done_pages, total_pages, f"Из кэша: {pdf.name}")

                records.append({"path": str(pdf.resolve()), "name": pdf.name, "pages": pages})

            if not records:
                raise RuntimeError("Ни один документ не удалось обработать.\n" + "\n".join(skipped_files[:10]))
            self.finished_ok.emit(records, len(pdf_files), len(skipped_files))
        except Exception as e:
            self.failed.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1320, 860)
        self.folder = None
        self.records = []
        self.matches = []
        self.worker = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        self.folder_btn = QPushButton("Выбрать папку с PDF…")
        self.folder_btn.clicked.connect(self.choose_folder)
        self.folder_label = QLabel("Папка не выбрана")
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.folder_btn)
        top.addWidget(self.folder_label, 1)
        layout.addLayout(top)

        self.status = QLabel("Выберите папку, в которой находятся все выпуски PDF.")
        layout.addWidget(self.status)

        actions = QHBoxLayout()
        self.ocr_btn = QPushButton("Распознать все документы")
        self.ocr_btn.clicked.connect(self.start_ocr)
        self.ocr_btn.setEnabled(False)
        actions.addWidget(self.ocr_btn)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        actions.addWidget(self.progress, 1)
        layout.addLayout(actions)

        searchrow = QHBoxLayout()
        searchrow.addWidget(QLabel("Искать слово или фразу:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('Например: "Шабалинское церковное училище"')
        self.search_input.returnPressed.connect(self.search)
        self.search_input.setEnabled(False)
        searchrow.addWidget(self.search_input, 1)
        self.search_btn = QPushButton("Найти во всех документах")
        self.search_btn.clicked.connect(self.search)
        self.search_btn.setEnabled(False)
        searchrow.addWidget(self.search_btn)
        layout.addLayout(searchrow)

        help_label = QLabel("Можно вводить одно слово или несколько слов подряд. Фраза ищется как последовательность соседних слов; знаки препинания не учитываются.")
        help_label.setStyleSheet("color: #555;")
        layout.addWidget(help_label)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Результаты поиска (документ / страница):"))
        self.results = QListWidget()
        self.results.currentRowChanged.connect(self.show_result)
        left_layout.addWidget(self.results)
        self.result_count = QLabel("Совпадений: 0")
        left_layout.addWidget(self.result_count)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_title = QLabel("Предпросмотр страницы")
        right_layout.addWidget(self.preview_title)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.image_label = QLabel("После выбора результата здесь появится страница с выделенным словом или фразой.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)
        right_layout.addWidget(self.scroll, 1)
        splitter.addWidget(right)
        splitter.setSizes([420, 850])
        layout.addWidget(splitter, 1)

        note = QLabel("Поиск выполняется по всем PDF выбранной папки и подпапок. Переносы через дефис учитываются. Распознанные страницы кэшируются на компьютере.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с PDF-документами")
        if not folder:
            return
        self.folder = folder
        self.folder_label.setText(folder)
        self.records = []
        self.matches = []
        self.results.clear()
        self.ocr_btn.setEnabled(True)
        self.search_input.setEnabled(False)
        self.search_btn.setEnabled(False)
        self.image_label.setText("Нажмите «Распознать все документы».")
        count = sum(1 for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
        self.status.setText(f"Найдено PDF-файлов: {count}. Нажмите «Распознать все документы».")

    def start_ocr(self):
        if not self.folder:
            return
        try:
            if "rus" not in pytesseract.get_languages(config=""):
                raise RuntimeError("Не найден русский языковой пакет rus.traineddata.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка OCR", str(e))
            return
        self.ocr_btn.setEnabled(False)
        self.folder_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.status.setText("Подсчёт страниц и подготовка к распознаванию…")
        self.worker = FolderOCRWorker(self.folder)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_ocr_done)
        self.worker.failed.connect(self.on_ocr_failed)
        self.worker.start()

    def on_progress(self, done, total, message):
        self.progress.setValue(int(done * 100 / max(total, 1)))
        self.status.setText(f"{message} — обработано страниц {done}/{total}")

    def on_ocr_done(self, records, total_files, skipped):
        self.records = records
        self.ocr_btn.setEnabled(True)
        self.folder_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.search_input.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.status.setText(
            f"Готово. Документов обработано: {len(records)} из {total_files}. "
            f"С ошибками пропущено: {skipped}. Поиск доступен."
        )
        self.search_input.setFocus()

    def on_ocr_failed(self, error):
        self.ocr_btn.setEnabled(True)
        self.folder_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Ошибка распознавания", error)
        self.status.setText("Не удалось обработать папку.")

    def search(self):
        query = self.search_input.text().strip()
        if not query or not query_tokens(query):
            QMessageBox.information(self, "Поиск", "Введите слово или фразу.")
            return
        if not self.records:
            QMessageBox.information(self, "Поиск", "Сначала распознайте документы.")
            return

        self.results.clear()
        self.matches = []
        for record in self.records:
            for page in record["pages"]:
                for found in find_matches(page["words"], query):
                    self.matches.append((record, page, found))

        for record, page, found in self.matches:
            self.results.addItem(QListWidgetItem(
                f"{record['name']} | стр. {page['page'] + 1}: {found['text']} (OCR {found['conf']:.0f}%)"
            ))
        self.result_count.setText(f"Совпадений: {len(self.matches)}")
        if self.matches:
            self.results.setCurrentRow(0)
        else:
            self.image_label.setText("Совпадений не найдено. Проверьте написание или попробуйте более короткую фразу.")

    def show_result(self, row):
        if row < 0 or row >= len(self.matches):
            return
        record, page_data, found = self.matches[row]
        try:
            doc = fitz.open(record["path"])
            page = doc.load_page(page_data["page"])
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            doc.close()
            qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
            painter = QPainter(qimg)
            painter.setPen(QPen(QColor(255, 35, 35), 5))
            for box in found["boxes"]:
                painter.drawRect(box["left"], box["top"], box["width"], box["height"])
            painter.end()
            pm = QPixmap.fromImage(qimg)
            max_width = max(600, self.scroll.viewport().width() - 30)
            if pm.width() > max_width:
                pm = pm.scaledToWidth(max_width, Qt.SmoothTransformation)
            self.image_label.setPixmap(pm)
            self.image_label.resize(pm.size())
            self.preview_title.setText(
                f"{record['name']} — страница {page_data['page'] + 1} — найдено: «{found['text']}»"
            )
        except Exception as e:
            QMessageBox.warning(self, "Ошибка открытия страницы", str(e))

def main():
    configure_bundled_tesseract()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
