import os
import random
import string
from pathlib import Path
from typing import List, Tuple
import io

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont


class SyntheticOCRTextGenerator:
    def __init__(self, output_dir: str = "dataset_crops", font_paths: List[str] = None):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Набор шрифтов: если пути не переданы, ищутся системные или используется дефолтный
        self.font_paths = font_paths or self._find_system_fonts()
        
        # Словари для генерации текста
        self.ru_alphabet = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
        self.en_alphabet = string.ascii_letters
        self.digits_symbols = string.digits + " .,!?-+=$%#@:;№/()"
        
        # Шаблоны генерации текстовых строк
        self.words_ru = [
            "Авито", "Объявление", "Цена", "Доставка", "Скидка", "Состояние", "В наличии",
            "Телефон", "Продажа", "Новый", "Б/У", "Гарантия", "Москва", "Самовывоз",
            "Оригинал", "Комплект", "Ремонт", "Услуги", "Отправка", "Быстро"
        ]
        self.words_en = [
            "Avito", "Sale", "Price", "Original", "New", "Delivery", "Discount",
            "Condition", "In stock", "Phone", "iPhone", "Samsung", "Nike", "Adidas",
            "Model", "Serial", "Brand", "Store", "Free", "Quality"
        ]

    def _find_system_fonts(self) -> List[str]:
        """Поиск доступных системных шрифтов в ОС."""
        candidate_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "C:\\Windows\\Fonts\\calibri.ttf",
            "C:\\Windows\\Fonts\\times.ttf",
            "/System/Library/Fonts/Helvetica.ttc"
        ]
        available = [p for p in candidate_paths if os.path.exists(p)]
        return available

    def _generate_text_string(self) -> str:
        """Формирует случайную текстовую строку (RU, EN, цифры, комбинации)."""
        mode = random.choice(["ru_word", "en_word", "price", "mixed", "random_chars"])
        
        if mode == "ru_word":
            return random.choice(self.words_ru)
        elif mode == "en_word":
            return random.choice(self.words_en)
        elif mode == "price":
            price = f"{random.randint(100, 500000):,}".replace(",", " ")
            curr = random.choice(["руб.", "₽", "руб", "$", "€"])
            return f"{price} {curr}"
        elif mode == "mixed":
            return f"{random.choice(self.words_en)} {random.choice(self.words_ru)}"
        else:
            length = random.randint(3, 12)
            chars = self.ru_alphabet + self.en_alphabet + string.digits
            return "".join(random.choices(chars, k=length))

    def _create_background(self, width: int, height: int) -> Image.Image:
        """Генерирует случайный фон: монотонный, градиент или шумную текстуру."""
        bg_type = random.choice(["solid", "gradient", "noise", "pattern"])
        
        if bg_type == "solid":
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            return Image.new("RGB", (width, height), color)
            
        elif bg_type == "gradient":
            c1 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
            c2 = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.float32)
            
            # Вертикальный или горизонтальный градиент
            if random.random() > 0.5:
                arr = np.linspace(c1, c2, width)
                arr = np.tile(arr, (height, 1, 1))
            else:
                arr = np.linspace(c1, c2, height)
                arr = np.tile(arr[:, np.newaxis, :], (1, width, 1))
            return Image.fromarray(arr.astype(np.uint8))
            
        elif bg_type == "noise":
            noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(noise)
            return img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
            
        else:  # pattern/texture
            base_color = np.array([random.randint(50, 200) for _ in range(3)])
            noise = np.random.randint(-30, 30, (height, width, 3))
            arr = np.clip(base_color + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)

    def _get_contrasting_text_color(self, bg_image: Image.Image) -> Tuple[int, int, int]:
        """Подбирает контрастный цвет текста относительно фона."""
        avg_color = np.array(bg_image).mean()
        if avg_color > 127:
            # Темный цвет для светлого фона
            return (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))
        else:
            # Светлый цвет для темного фона
            return (random.randint(180, 255), random.randint(180, 255), random.randint(180, 255))

    def _apply_augmentations(self, image: Image.Image) -> Image.Image:
        """Добавляет артефакты реального OCR-детектора: размытие, шум, JPEG-сжатие (Pure PIL)."""

        # 1. Небольшой случайный наклон (до ±5°), имитирующий неидеальный бокс детектора
        if random.random() < 0.6:
            angle = random.uniform(-5.0, 5.0)
            image = image.rotate(
                angle, resample=Image.Resampling.BICUBIC, expand=False
            )

        # 2. Гауссовское размытие / расфокусировка
        if random.random() < 0.4:
            radius = random.uniform(0.5, 1.5)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # 3. Имитация артефактов JPEG-сжатия через буфер в памяти
        if random.random() < 0.5:
            quality = random.randint(15, 70)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")

        return image

    def generate_single_crop(self, crop_id: int) -> dict:
        """Генерирует один кроп и возвращает метку ориентации."""
        text = self._generate_text_string()
        font_size = random.randint(18, 64)
        
        # Выбор шрифта
        if self.font_paths:
            font_path = random.choice(self.font_paths)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()

        # Вычисление размеров текста с учетом отступов (padding)
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = max(bbox[2] - bbox[0], 10)
        text_h = max(bbox[3] - bbox[1], 10)

        padding_x = random.randint(6, 25)
        padding_y = random.randint(4, 18)
        crop_w = text_w + 2 * padding_x
        crop_h = text_h + 2 * padding_y

        # Генерация фона и цвета текста
        bg = self._create_background(crop_w, crop_h)
        text_color = self._get_contrasting_text_color(bg)
        
        # Отрисовка текста
        draw = ImageDraw.Draw(bg)
        text_x = padding_x - bbox[0]
        text_y = padding_y - bbox[1]
        
        # Опциональная обводка/тень текста
        if random.random() < 0.3:
            stroke_width = random.randint(1, 2)
            stroke_color = (255 - text_color[0], 255 - text_color[1], 255 - text_color[2])
            draw.text((text_x, text_y), text, fill=text_color, font=font, 
                      stroke_width=stroke_width, stroke_fill=stroke_color)
        else:
            draw.text((text_x, text_y), text, fill=text_color, font=font)

        # Применение аугментаций до поворота
        crop_img = self._apply_augmentations(bg)

        # Случайный поворот на 180 градусов (Target p_180)
        is_180 = random.choice([0, 1])
        if is_180 == 1:
            crop_img = crop_img.rotate(180, expand=False)

        # Сохранение файла
        file_name = f"crop_{crop_id:06d}.jpg"
        save_path = self.images_dir / file_name
        crop_img.save(save_path, quality=95)

        return {
            "image_path": str(save_path.relative_to(self.output_dir)),
            "text_gt": text,
            "is_180": is_180
        }

    def generate_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """Генерирует датасет заданного размера и сохраняет metadata.csv."""
        records = []
        print(f"Старт генерации {num_samples} кропов...")
        
        for i in range(num_samples):
            record = self.generate_single_crop(i)
            records.append(record)
            if (i + 1) % 500 == 0 or (i + 1) == num_samples:
                print(f"Сгенерировано {i + 1}/{num_samples} образцов")

        df = pd.DataFrame(records)
        csv_path = self.output_dir / "labels.csv"
        df.to_csv(csv_path, index=False)
        print(f"Датасет успешно сохранен в directory: {self.output_dir.resolve()}")
        print(f"Разметка сохранена в: {csv_path.resolve()}")
        return df


if __name__ == "__main__":
    # Пример использования
    # При необходимости укажите явные пути к .ttf шрифтам
    custom_fonts = [
        # "/path/to/Arial.ttf",
        # "/path/to/Roboto-Regular.ttf"
    ]
    
    generator = SyntheticOCRTextGenerator(
        output_dir="synthetic_ocr_dataset",
        font_paths=custom_fonts if custom_fonts else None
    )
    
    # Сгенерировать 5,000 синтетических кропов
    df_labels = generator.generate_dataset(num_samples=5000)
    print(df_labels.head())