import sys
import os
import json
import hashlib
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

APP_NAME = "Поиск по PDF"
CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "PDFWordSearch" / "cache"

def resource_path(*parts):
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return root.joinpath(*parts)

def configure_bundled_tesseract():
    # The GitHub Actions build bundles Tesseract and Russian traineddata inside the EXE.
    tess_dir = resource_path("tesseract")
    exe = tess_dir / "tesseract.exe"
    data_dir = tess_dir / "tessdata"
    if exe.exists():
        pytesseract.pytesseract.tesseract_cmd = str(exe)
        os.environ["TESSDATA_PREFIX"] = str(data_dir)
        os.environ["PATH"] = str(tess_dir) + os.pathsep + os.environ.get("PATH", "")

def normalize(s: str) -> str:
    s = s.casefold().replace("ё", "е")
    return s.translate(str.maketrans({"ѣ": "е", "і": "и", "ѳ": "ф", "ѵ": "и"}))

class OCRWorker(QThread):
    progress = Signal(int, int)
    finished_ok = Signal(str, int)
    failed = Signal(str)

    def __init__(self, pdf_path, lang="rus"):
        super().__init__()
        self.pdf_path = str(pdf_path)
        self.lang = lang

    def run(self):
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            stat = Path(self.pdf_path).stat()
            digest = hashlib.sha256((self.pdf_path + str(stat.st_mtime_ns) + str(stat.st_size)).encode("utf-8")).hexdigest()[:20]
            cache_file = CACHE_DIR / f"{digest}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    self.finished_ok.emit(str(cache_file), len(data.get("pages", [])))
                    return
                except Exception:
                    cache_file.unlink(missing_ok=True)

            doc = fitz.open(self.pdf_path)
            pages = []
            total = len(doc)
            for page_no, page in enumerate(doc):
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                data = pytesseract.image_to_data(
                    image, lang=self.lang, config="--oem 1 --psm 6",
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
                        "conf": conf
                    })
                pages.append({"page": page_no, "width": pix.width, "height": pix.height, "words": words})
                self.progress.emit(page_no + 1, total)
            doc.close()
            cache_file.write_text(json.dumps({"pdf_path": self.pdf_path, "pages": pages}, ensure_ascii=False), encoding="utf-8")
            self.finished_ok.emit(str(cache_file), total)
        except Exception as e:
            self.failed.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1180, 820)
        self.pdf_path = None
        self.index_data = None
        self.matches = []
        self.worker = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        self.open_btn = QPushButton("Открыть PDF…")
        self.open_btn.clicked.connect(self.open_pdf)
        self.file_label = QLabel("PDF не выбран")
        self.file_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.open_btn)
        top.addWidget(self.file_label, 1)
        layout.addLayout(top)

        self.status = QLabel("Выберите сканированный PDF и нажмите «Распознать текст».")
        layout.addWidget(self.status)

        actions = QHBoxLayout()
        self.ocr_btn = QPushButton("Распознать текст")
        self.ocr_btn.clicked.connect(self.start_ocr)
        self.ocr_btn.setEnabled(False)
        actions.addWidget(self.ocr_btn)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        actions.addWidget(self.progress, 1)
        layout.addLayout(actions)

        searchrow = QHBoxLayout()
        searchrow.addWidget(QLabel("Искать слово или часть слова:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Например: Тобольск")
        self.search_input.returnPressed.connect(self.search)
        self.search_input.setEnabled(False)
        searchrow.addWidget(self.search_input, 1)
        self.search_btn = QPushButton("Найти")
        self.search_btn.clicked.connect(self.search)
        self.search_btn.setEnabled(False)
        searchrow.addWidget(self.search_btn)
        layout.addLayout(searchrow)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Результаты поиска:"))
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
        self.image_label = QLabel("После поиска здесь появится страница с выделенным словом.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll.setWidget(self.image_label)
        right_layout.addWidget(self.scroll, 1)
        splitter.addWidget(right)
        splitter.setSizes([360, 800])
        layout.addWidget(splitter, 1)

        note = QLabel("Для старинной печати OCR может ошибаться. Если совпадение не найдено, попробуйте часть слова или другой вариант написания.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #666;")
        layout.addWidget(note)

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите PDF", "", "PDF-файлы (*.pdf)")
        if not path:
            return
        self.pdf_path = path
        self.file_label.setText(path)
        self.ocr_btn.setEnabled(True)
        self.search_input.setEnabled(False)
        self.search_btn.setEnabled(False)
        self.results.clear()
        self.image_label.setText("Нажмите «Распознать текст».")
        self.status.setText("PDF выбран. Для сканов сначала требуется распознавание текста.")
        self.index_data = None

    def start_ocr(self):
        if not self.pdf_path:
            return
        try:
            langs = pytesseract.get_languages(config="")
            if "rus" not in langs:
                raise RuntimeError("В автономную сборку не включён русский языковой пакет rus.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка OCR", f"Не удалось запустить встроенное распознавание текста.\n\n{e}")
            return
        self.ocr_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.status.setText("Распознавание страниц… Это может занять несколько минут.")
        self.worker = OCRWorker(self.pdf_path)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_ocr_done)
        self.worker.failed.connect(self.on_ocr_failed)
        self.worker.start()

    def on_progress(self, done, total):
        self.progress.setMaximum(total)
        self.progress.setValue(done)
        self.status.setText(f"Распознано страниц: {done} из {total}")

    def on_ocr_done(self, cache_file, pages):
        self.index_data = json.loads(Path(cache_file).read_text(encoding="utf-8"))
        self.ocr_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.progress.setVisible(False)
        self.search_input.setEnabled(True)
        self.search_btn.setEnabled(True)
        self.status.setText(f"Готово: обработано страниц — {pages}. Введите запрос.")
        self.search_input.setFocus()

    def on_ocr_failed(self, error):
        self.ocr_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Ошибка распознавания", error)
        self.status.setText("Не удалось распознать PDF.")

    def search(self):
        if not self.index_data:
            return
        query = normalize(self.search_input.text().strip())
        if not query:
            QMessageBox.information(self, "Поиск", "Введите слово или часть слова.")
            return
        self.results.clear()
        self.matches = []
        for page in self.index_data["pages"]:
            for word in page["words"]:
                if query in normalize(word["text"]):
                    self.matches.append((page, word))
        for page, word in self.matches:
            self.results.addItem(QListWidgetItem(f"Стр. {page['page'] + 1}: {word['text']}  (OCR: {word['conf']:.0f}%)"))
        self.result_count.setText(f"Совпадений: {len(self.matches)}")
        if self.matches:
            self.results.setCurrentRow(0)
        else:
            self.image_label.setText("Совпадений не найдено. Попробуйте часть слова или другой вариант написания.")

    def show_result(self, row):
        if row < 0 or row >= len(self.matches):
            return
        page_data, word = self.matches[row]
        doc = fitz.open(self.pdf_path)
        page = doc.load_page(page_data["page"])
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        doc.close()
        qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
        painter = QPainter(qimg)
        painter.setPen(QPen(QColor(255, 35, 35), 5))
        painter.drawRect(word["left"], word["top"], word["width"], word["height"])
        painter.end()
        pm = QPixmap.fromImage(qimg)
        max_width = max(600, self.scroll.viewport().width() - 30)
        if pm.width() > max_width:
            pm = pm.scaledToWidth(max_width, Qt.SmoothTransformation)
        self.image_label.setPixmap(pm)
        self.image_label.resize(pm.size())
        self.preview_title.setText(f"Страница {page_data['page'] + 1} — найдено: «{word['text']}»")

def main():
    configure_bundled_tesseract()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
