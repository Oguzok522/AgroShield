import asyncio
import csv
import io
import os
import random
import re
import string
import logging
from datetime import datetime, timedelta
import folium
import numpy as np
import aiohttp
import aiosqlite
import tensorflow as tf
from PIL import Image, ExifTags
from PIL.ExifTags import GPSTAGS
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import FSInputFile
from dotenv import load_dotenv
from folium.plugins import HeatMap
from pillow_heif import register_heif_opener

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

register_heif_opener()
load_dotenv()

TOKEN = os.getenv('BOT_TOKEN')
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
DB_FILE = 'agro_final.sqlite'

CLASS_NAMES = [
    "Здоровый картофель",
    "Раняя сухая пятнистость(Альтернариоз)",
    "Позднее увядание(Фитофтороз)"
]

# Загружаем модель один раз при старте
try:
    model_potato = tf.keras.models.load_model('agroshield_m.keras')
except Exception as e:
    logger.error(f"Ошибка загрузки модели: {e}")

bot = Bot(TOKEN)
dp = Dispatcher()

# ОБРАБОТКА ДАННЫХ И ИНФЕРЕНС НЕЙРОСЕТИ


def dms_to_decimal(dms, reference):
    """Конвертация координат из градусов/минут/секунд в десятичный формат"""
    degrees, minutes, seconds = dms[0], dms[1], dms[2]
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if reference in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_gps(img):
    """Извлечение скрытых бинарных метаданных EXIF напрямую из матрицы фото"""
    try:
        exif = img.getexif()
        if not exif:
            return None
        gps_ifd = exif.get_ifd(0x8825)
        if not gps_ifd:
            return None

        gps_tags = {GPSTAGS.get(key, key): val for key, val in gps_ifd.items()}

        if 'GPSLatitude' in gps_tags and 'GPSLongitude' in gps_tags:
            lat = dms_to_decimal(gps_tags['GPSLatitude'], gps_tags.get('GPSLatitudeRef', 'N'))
            lon = dms_to_decimal(gps_tags['GPSLongitude'], gps_tags.get('GPSLongitudeRef', 'E'))
            if lat is not None and lon is not None:
                return lat, lon
        return None
    except Exception as e:
        logger.warning(f"Ошибка извлечения GPS: {e}")
        return None

def process_image(image_bytes):
    """
    Оптимизированный инференс (training=False).
    Обработка изображения в оперативной памяти (BytesIO) без сохранения на диск.
    """
    img = Image.open(io.BytesIO(image_bytes))
    coordinates = extract_gps(img)

    img = img.convert('RGB').resize((224, 224))
    img_array = tf.expand_dims(tf.keras.utils.img_to_array(img), 0)

    predictions = model_potato(img_array, training=False).numpy()
    class_index = int(np.argmax(predictions))
    confidence = float(np.max(predictions) * 100)

    return CLASS_NAMES[class_index], confidence, coordinates

#  АРХИТЕКТУРА БАЗЫ ДАННЫХ И МАРШРУТИЗАЦИЯ

async def init_db():
    """
    Инициализация реляционной БД.
    Реализована мягкая миграция (try-except) для столбца status без потери старых данных.
    """
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                enterprise TEXT,
                timestamp DATETIME,
                diagnosis TEXT,
                confidence REAL,
                latitude REAL,
                longitude REAL,
                status TEXT DEFAULT 'active'
            )
        ''')
        # Мягкая миграция БД
        try:
            await db.execute("ALTER TABLE diagnostics ADD COLUMN status TEXT DEFAULT 'active'")
        except aiosqlite.OperationalError:
            pass

        # Архитектура Multi-tenancy (Мультиарендность)
        await db.execute('''
            CREATE TABLE IF NOT EXISTS enterprises (
                name TEXT PRIMARY KEY,
                invite_code TEXT,
                chat_id INTEGER
            )
        ''')
        await db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                enterprise TEXT
            )
        ''')
        await db.commit()

async def register_enterprise(name, code, chat_id):
    """Идемпотентное сохранение предприятия с защитой от SQL-инъекций (?,?,?)"""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('INSERT OR REPLACE INTO enterprises (name, invite_code, chat_id) VALUES (?, ?, ?)',
                         (name, code, chat_id))
        await db.commit()

async def get_enterprise_by_code(code):
    """Поиск предприятия по коду с использованием безопасного извлечения (тернарный оператор)"""
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute('SELECT name FROM enterprises WHERE invite_code = ?', (code,)) as cursor:
            res = await cursor.fetchone()
            return res[0] if res else None

async def bind_user(user_id, enterprise):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('INSERT OR REPLACE INTO users (user_id, enterprise) VALUES (?, ?)', (user_id, enterprise))
        await db.commit()

async def get_user_enterprise(user_id):
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute('SELECT enterprise FROM users WHERE user_id = ?', (user_id,)) as cursor:
            res = await cursor.fetchone()
            return res[0] if res else None

async def get_dispatcher_chat(enterprise):
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute('SELECT chat_id FROM enterprises WHERE name = ?', (enterprise,)) as cursor:
            res = await cursor.fetchone()
            return res[0] if res else None

async def save_diagnostic(user_id, enterprise, diagnosis, confidence, coordinates):
    """Сохранение инцидента с автоматическим присвоением статуса 'active'"""
    lat = coordinates[0] if coordinates else None
    lon = coordinates[1] if coordinates else None
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute('''
            INSERT INTO diagnostics (user_id, enterprise, timestamp, diagnosis, confidence, latitude, longitude, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active')
        ''', (user_id, enterprise, current_time, diagnosis, confidence, lat, lon))
        await db.commit()

async def get_today_active_diagnostics(enterprise_name):
    """
    ОПЕРАТИВНАЯ СВОДКА: Гибридная фильтрация данных (SQL WHERE + Python timedelta).
    Возвращает очаги строго за последние 24 часа для планирования текущего дня.
    """
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
            """
            SELECT id, diagnosis, latitude, longitude, timestamp
            FROM diagnostics
            WHERE latitude IS NOT NULL AND enterprise = ? AND status = 'active'
            ORDER BY id ASC
            """, (enterprise_name,)
        ) as cursor:
            rows = await cursor.fetchall()

    today_rows = []
    one_day_ago = datetime.now() - timedelta(days=1)

    for row in rows:
        rec_id, diag, lat, lon, timestamp_str = row
        try:
            record_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            if record_time >= one_day_ago:
                today_rows.append(row)
        except ValueError:
            continue

    return today_rows

async def hide_diagnostic_from_map(record_id, enterprise_name):
    """Паттерн Soft Delete: визуальное удаление с карты с сохранением в БД для статистики"""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("UPDATE diagnostics SET status = 'hidden' WHERE id = ? AND enterprise = ?",
                         (record_id, enterprise_name))
        await db.commit()

async def hide_by_exact_coordinates(lat, lon, enterprise_name):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            """
            UPDATE diagnostics SET status = 'hidden'
            WHERE enterprise = ? AND ROUND(latitude, 5) = ROUND(?, 5) AND ROUND(longitude, 5) = ROUND(?, 5)
            """, (enterprise_name, lat, lon)
        )
        await db.commit()

async def delete_by_exact_coordinates(lat, lon, enterprise_name):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute(
            """
            DELETE FROM diagnostics
            WHERE enterprise = ? AND ROUND(latitude, 5) = ROUND(?, 5) AND ROUND(longitude, 5) = ROUND(?, 5)
            """, (enterprise_name, lat, lon)
        )
        await db.commit()

async def get_statistics(enterprise_name, period="all"):
    async with aiosqlite.connect(DB_FILE) as db:
        if period == "today":
            today_str = datetime.now().strftime("%Y-%m-%d")
            async with db.execute(
                """
                SELECT diagnosis, COUNT(*) FROM diagnostics
                WHERE enterprise = ? AND timestamp LIKE ?
                GROUP BY diagnosis
                """, (enterprise_name, f"{today_str}%")
            ) as cursor:
                return await cursor.fetchall()
        else:
            async with db.execute(
                "SELECT diagnosis, COUNT(*) FROM diagnostics WHERE enterprise = ? GROUP BY diagnosis",
                (enterprise_name,)
            ) as cursor:
                return await cursor.fetchall()


#  ПОГОДА И ГЕО-АНАЛИТИКА

async def calculate_enterprise_center(enterprise_name):
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
            "SELECT latitude, longitude FROM diagnostics WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND enterprise = ?",
            (enterprise_name,)
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows: return None
    return sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows)

async def fetch_weather_forecast_for_enterprise(enterprise_name):
    center = await calculate_enterprise_center(enterprise_name)
    if not center:
        return "⚠️ У предприятия пока нет оцифрованных очагов с GPS-координатами для метеоанализа."

    lat, lon = center
    if not WEATHER_API_KEY:
        return "❌ Не настроен API-ключ OpenWeatherMap (проверьте файл .env)."

    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=ru"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    forecast_list = data['list'][:24]

                    max_humidity = max(item['main']['humidity'] for item in forecast_list)
                    avg_temp = sum(item['main']['temp'] for item in forecast_list) / len(forecast_list)

                    report = (
                        f"🌍 <b>ПРЕДИКТИВНЫЙ АНАЛИЗ ПОГОДЫ (ХОЗЯЙСТВО: {enterprise_name})</b>\n"
                        f"📍 Геоцентр полей: <code>{lat:.5f}, {lon:.5f}</code>\n"
                        f"🌡 Средняя t° (3 дня): <b>{avg_temp:.1f}°C</b> | Пиковая влажность: <b>{max_humidity}%</b>\n\n"
                    )

                    if max_humidity >= 85 and 15 <= avg_temp <= 22:
                        report += (
                            "🚨 <b>ВНИМАНИЕ: КРИТИЧЕСКИЙ РИСК ЭПИФИТОТИИ!</b>\n"
                            "Условия в географическом центре хозяйства идеальны для фитофтороза. Рекомендуется выезд техники."
                        )
                    else:
                        report += "🟢 <b>УГРОЗА УМЕРЕННАЯ:</b>\nУсловия стабильные. Продолжайте мониторинг."
                    return report
                return "❌ Ошибка получения данных от метеосервиса."
    except aiohttp.ClientError as e:
        logger.error(f"Ошибка запроса погоды: {e}")
        return "❌ Сбой сетевого соединения с метеосервером."


#  ГЕНЕРАЦИЯ ОТЧЕТОВ И ТЕПЛОВЫХ КАРТ

def clean_name(text):
    return re.sub(r'[^\w\-]', '_', text)

async def export_to_csv_async(enterprise_name, period="all"):
    async with aiosqlite.connect(DB_FILE) as db:
        if period == "today":
            today_str = datetime.now().strftime("%Y-%m-%d")
            async with db.execute(
                """
                SELECT id, timestamp, diagnosis, confidence, latitude, longitude, status
                FROM diagnostics
                WHERE enterprise = ? AND timestamp LIKE ?
                """, (enterprise_name, f"{today_str}%")
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with db.execute(
                "SELECT id, timestamp, diagnosis, confidence, latitude, longitude, status FROM diagnostics WHERE enterprise = ?",
                (enterprise_name,)
            ) as cursor:
                rows = await cursor.fetchall()

    if not rows: return None

    filename = f"agro_report_{period}_{clean_name(enterprise_name)}.csv"
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['ID', 'Дата и время', 'Диагноз', 'Уверенность ИИ (%)', 'Широта', 'Долгота', 'Статус (active/hidden)'])
        writer.writerows(rows)
    return filename

def generate_heatmap_from_db(enterprise_name):
    """
    СТРАТЕГИЧЕСКАЯ ТЕПЛОВАЯ КАРТА:
    Строится по принципу Soft Delete. Агрегирует ВСЕ инциденты со статусом 'active',
    независимо от времени обнаружения, пока агроном не подтвердит обработку поля.
    """
    import sqlite3
    center_lat, center_lon = 53.8058, 27.5255
    alt_coords, phyto_coords, markers_data = [], [], []
    last_point = None

    try:
        with sqlite3.connect(DB_FILE) as conn:
            rows = conn.execute(
                "SELECT latitude, longitude, confidence, diagnosis FROM diagnostics WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND diagnosis != 'Здоровый картофель' AND enterprise = ? AND status = 'active'",
                (enterprise_name,)
            ).fetchall()

            for lat, lon, conf, diag in rows:
                weight = max(0.1, min(1.0, (conf or 100) / 100))
                if "Альтернариоз" in diag:
                    alt_coords.append([lat, lon, weight])
                else:
                    phyto_coords.append([lat, lon, weight])

                markers_data.append((lat, lon, diag))
                last_point = (lat, lon)
    except Exception as e:
        logger.error(f"Ошибка чтения БД для карты: {e}")

    if last_point:
        center_lat, center_lon = last_point[0], last_point[1]

    base_map = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")
    base_map.get_root().header.add_child(
        folium.Element("<style>.leaflet-control-attribution svg { display: none !important; }</style>"))

    if alt_coords: HeatMap(alt_coords, radius=15, blur=10, gradient={0.4: 'yellow', 1.0: 'orange'}).add_to(base_map)
    if phyto_coords: HeatMap(phyto_coords, radius=15, blur=10, gradient={0.4: 'orange', 1.0: 'red'}).add_to(base_map)

    for lat, lon, diag in markers_data:
        point_color = "yellow" if "Альтернариоз" in diag else "red"
        folium.CircleMarker(
            location=[lat, lon], radius=5, weight=1, color=point_color,
            fill=True, fill_color=point_color, fill_opacity=0.6,
            popup=f"<b>{diag}</b><br>Координаты: {lat}, {lon}",
            tooltip=f"<b>{diag}</b><br>📍 {lat}, {lon}"
        ).add_to(base_map)

    output_filename = f"map_{clean_name(enterprise_name)}.html"
    base_map.save(output_filename)
    return output_filename


#КЛАВИАТУРЫ И ХЕНДЛЕРЫ TELEGRAM

def get_action_keyboard(disease_name, coordinates):
    if coordinates:
        lat, lon = coordinates
        risk_callback = f"point_risk_{lat:.2f}_{lon:.2f}"
    else:
        risk_callback = "point_risk_no_geo"

    buttons = [
        [types.InlineKeyboardButton(text="🚨 Отправить в Ситуационный центр", callback_data="send_incident")],
        [types.InlineKeyboardButton(text="☁️ Прогноз погоды для этого очага", callback_data=risk_callback)],
        [types.InlineKeyboardButton(text="💊 Рекомендации по лечению", callback_data=f"treat_{disease_name[:15]}")]
    ]
    return types.InlineKeyboardMarkup(inline_keyboard=buttons)

@dp.message(CommandStart())
async def hello(message: types.Message):
    main_markup = types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text="🗺 Показать тепловую карту полей")],
            [types.KeyboardButton(text="📝 Управление картой"), types.KeyboardButton(text="📥 Отчеты и Статистика")],
            [types.KeyboardButton(text="☁️ Анализ погоды и рисков")],
            [types.KeyboardButton(text="❓ Инструкция")]
        ],
        resize_keyboard=True
    )
    await message.answer(
        "Привет! Я AI-диагност АгроЩит. 🌿\n\n"
        "Отправь мне фото листа как <b>ФАЙЛ (Документ)</b>, чтобы я смог прочитать GPS-координаты.\n"
        "Либо отправь как обычное фото для быстрой проверки без карты.\n\n"
        "Обязательно ознакомься с <b>❓ Инструкцией</b> перед началом работы!",
        parse_mode="HTML",
        reply_markup=main_markup
    )

@dp.message(F.text == "❓ Инструкция")
async def send_help(message: types.Message):
    help_text = (
        "📖 <b>ПОЛНОЕ РУКОВОДСТВО ПОЛЬЗОВАТЕЛЯ</b>\n\n"
        "<b>1. РЕГИСТРАЦИЯ И АВТОРИЗАЦИЯ</b>\n"
        "• <b>Создать предприятие:</b> добавьте бота в общую рабочую группу Telegram и отправьте команду <code>/bind [Название хозяйства]</code>. Бот выдаст секретный AGRO-код.\n"
        "• <b>Подключиться к предприятию:</b> скопируйте выданный AGRO-код и отправьте его боту сюда (в личные сообщения).\n\n"
        "<b>2. КАК ВКЛЮЧИТЬ ГЕОЛОКАЦИЮ И СДЕЛАТЬ ФОТО</b>\n"
        "Без точных координат бот не сможет нанести очаг болезни на карту полей.\n"
        "🍏 <b>iPhone:</b> Настройки ➔ Конфиденциальность и безопасность ➔ Службы геолокации ➔ Камера ➔ Выбрать «При использовании приложения».\n"
        "🤖 <b>Android:</b> Откройте приложение «Камера» ➔ Настройки (иконка шестеренки) ➔ Включите «Теги геолокации» (или «Сохранять местоположение»).\n"
        "• Сфотографируйте больной лист крупным планом.\n"
        "• Отправьте фото боту <b>СТРОГО КАК ФАЙЛ</b> (Иконка скрепки ➔ Файл ➔ Фото). Если отправить как обычное фото, координаты автоматически удалятся мессенджером!\n\n"
        "<b>3. РАБОТА С КАРТОЙ</b>\n"
        "• <b>Управление картой</b>: здесь показаны активные очаги за последние 24 часа. Кнопка «Показать на карте» перебросит вас в навигатор. Кнопка «✅ Вылечено» скроет очаг с карты, но оставит его в базе для отчетов.\n"
        "• <b>Тепловая карта</b>: генерирует карту активных зон болезни для всего предприятия.\n\n"
        "<b>4. ИСПРАВЛЕНИЕ ОШИБОК И АРХИВ</b>\n"
        "• <b>Скрыть старый очаг с карты</b> (сохранить в базе для истории):\n"
        "  <code>/hide_coord [широта] [долгота]</code>\n"
        "• <b>Удалить ошибочную точку</b> (стереть навсегда без следа):\n"
        "  <code>/del_coord [широта] [долгота]</code>\n"
        "  <i>Пример: /del_coord 53.80582 27.52554</i>\n\n"
        "<b>5. ОТЧЕТЫ</b>\n"
        "В разделе «Отчеты и Статистика» можно выгрузить Excel-файлы за текущий день или за всё время работы."
    )
    await message.answer(help_text, parse_mode="HTML")

@dp.message(F.text == "☁️ Анализ погоды и рисков")
async def weather_risk_handler(message: types.Message):
    user_ent = await get_user_enterprise(message.from_user.id)
    if not user_ent: return await message.answer("❌ Сначала привяжитесь к хозяйству с помощью AGRO-кода.")

    wait_msg = await message.answer("🛰 Запрос метеоданных по географическому центру полей...")
    forecast_report = await fetch_weather_forecast_for_enterprise(user_ent)
    await message.answer(forecast_report, parse_mode="HTML")
    await wait_msg.delete()

@dp.message(F.text.startswith("/bind "))
async def bind_enterprise_group(message: types.Message):
    if message.chat.type in ["group", "supergroup"]:
        enterprise_name = message.text.replace("/bind ", "").strip()
        if enterprise_name:
            secret_code = f"AGRO-{''.join(random.choices(string.digits, k=4))}"
            await register_enterprise(enterprise_name, secret_code, message.chat.id)
            await message.answer(
                f"✅ <b>Система АгроЩит активирована!</b>\n"
                f"Предприятие: <i>{enterprise_name}</i>\n"
                f"🔑 <b>Секретный код:</b> <code>{secret_code}</code>\n\n"
                f"💡 <b>Что делать дальше?</b>\n"
                f"Скопируйте этот код и передайте вашим сотрудникам/агрономам. "
                f"Они должны отправить его боту в <b>личные сообщения</b> для подключения к базе хозяйства.",
                parse_mode="HTML"
            )

@dp.message(F.text.startswith("AGRO-"))
async def join_enterprise(message: types.Message):
    if message.chat.type == "private":
        code = message.text.strip()
        ent_name = await get_enterprise_by_code(code)
        if ent_name:
            await bind_user(message.from_user.id, ent_name)
            await message.answer(f"✅ Успешно! Вы привязаны к предприятию: <b>{ent_name}</b>.", parse_mode="HTML")

@dp.message(F.text == "🗺 Показать тепловую карту полей")
async def send_heatmap(message: types.Message):
    user_ent = await get_user_enterprise(message.from_user.id)
    if not user_ent: return await message.answer("❌ Введите AGRO-код для привязки к предприятию!")

    msg = await message.answer("🔄 Генерация тепловой карты...")
    map_file = await asyncio.to_thread(generate_heatmap_from_db, user_ent)
    await message.answer_document(FSInputFile(map_file), caption=f"🌐 <b>Тепловая карта: {user_ent}</b>", parse_mode="HTML")
    await msg.delete()

@dp.message(F.text == "📝 Управление картой")
async def manage_map(message: types.Message):
    user_ent = await get_user_enterprise(message.from_user.id)
    if not user_ent: return await message.answer("❌ Введите AGRO-код для привязки к предприятию!")

    records = await get_today_active_diagnostics(user_ent)

    if not records:
        await message.answer(f"✅ За последние сутки (24 часа) в базе <b>{user_ent}</b> новых активных очагов нет.", parse_mode="HTML")
    else:
        await message.answer(f"<b>Актуальные очаги за сутки ({user_ent}):</b>", parse_mode="HTML")
        for idx, (rec_id, diag, lat, lon, timestamp) in enumerate(records, start=1):
            text = f"📍 <b>Очаг №{idx}</b> ({timestamp})\nДиагноз: {diag}\nКоординаты: <code>{lat}, {lon}</code>"
            kb = types.InlineKeyboardMarkup(
                inline_keyboard=[
                    [types.InlineKeyboardButton(text="🗺 Показать на карте", url=f"https://www.google.com/maps?q={lat},{lon}")],
                    [types.InlineKeyboardButton(text="✅ Вылечено (Убрать с карты)", callback_data=f"hide_{rec_id}")]
                ]
            )
            await message.answer(text, reply_markup=kb, parse_mode="HTML", disable_web_page_preview=True)

    await message.answer(
        "⚠️ <b>Управление архивными точками:</b>\n"
        "Для точек, которые уже пропали из этого списка, используйте команды с координатами:\n\n"
        "👁 <b>Убрать с карты (оставить в архиве):</b>\n"
        "<code>/hide_coord [широта] [долгота]</code>\n\n"
        "🗑 <b>Удалить из БД навсегда (ошибка):</b>\n"
        "<code>/del_coord [широта] [долгота]</code>",
        parse_mode="HTML"
    )

@dp.callback_query(F.data.startswith("hide_"))
async def call_hide_point(callback: types.CallbackQuery):
    rec_id = int(callback.data.split("_")[1])
    user_ent = await get_user_enterprise(callback.from_user.id)
    await hide_diagnostic_from_map(rec_id, user_ent)
    await callback.answer("✅ Очаг вылечен и убран с карты!", show_alert=True)
    await callback.message.edit_text(callback.message.html_text + "\n\n<i>(✅ Очаг вылечен и скрыт с карты)</i>", parse_mode="HTML", disable_web_page_preview=True)

@dp.message(F.text.startswith("/hide_coord "))
async def hide_coord_command(message: types.Message):
    user_ent = await get_user_enterprise(message.from_user.id)
    if not user_ent: return await message.answer("❌ Сначала привяжитесь к хозяйству.")

    parts = message.text.split()
    if len(parts) < 3: return await message.answer("❌ Формат: <code>/hide_coord 53.80582 27.52554</code>", parse_mode="HTML")

    try:
        lat, lon = float(parts[1]), float(parts[2])
    except ValueError:
        return await message.answer("❌ Координаты должны быть числами.")

    await hide_by_exact_coordinates(lat, lon, user_ent)
    await message.answer(f"✅ Точка <code>{lat}, {lon}</code> вылечена и скрыта с тепловой карты.", parse_mode="HTML")

@dp.message(F.text.startswith("/del_coord "))
async def delete_coord_command(message: types.Message):
    user_ent = await get_user_enterprise(message.from_user.id)
    if not user_ent: return await message.answer("❌ Сначала привяжитесь к хозяйству.")

    parts = message.text.split()
    if len(parts) < 3: return await message.answer("❌ Формат: <code>/del_coord 53.80582 27.52554</code>", parse_mode="HTML")

    try:
        lat, lon = float(parts[1]), float(parts[2])
    except ValueError:
        return await message.answer("❌ Координаты должны быть числами.")

    await delete_by_exact_coordinates(lat, lon, user_ent)
    await message.answer(f"🗑 Точка <code>{lat}, {lon}</code> безвозвратно удалена из базы данных.", parse_mode="HTML")

@dp.message(F.text == "📥 Отчеты и Статистика")
async def show_statistics_menu(message: types.Message):
    user_ent = await get_user_enterprise(message.from_user.id)
    if not user_ent: return await message.answer("❌ Введите AGRO-код для привязки к предприятию!")

    kb = types.InlineKeyboardMarkup(
        inline_keyboard=[
            [types.InlineKeyboardButton(text="📊 Сводка за СЕГОДНЯ", callback_data="stats_today")],
            [types.InlineKeyboardButton(text="📈 Сводка за ВСЕ ВРЕМЯ", callback_data="stats_all")],
            [types.InlineKeyboardButton(text="📥 Скачать Excel (Сегодня)", callback_data="excel_today")],
            [types.InlineKeyboardButton(text="📥 Скачать Excel (Все время)", callback_data="excel_all")]
        ]
    )
    await message.answer(f"Выберите тип отчета для хозяйства <b>{user_ent}</b>:", reply_markup=kb, parse_mode="HTML")

@dp.callback_query(F.data.startswith("stats_"))
async def process_stats(callback: types.CallbackQuery):
    user_ent = await get_user_enterprise(callback.from_user.id)
    period = callback.data.split("_")[1]

    stats = await get_statistics(user_ent, period)
    title = "СЕГОДНЯ" if period == "today" else "ВСЕ ВРЕМЯ"

    if not stats: return await callback.message.answer(f"📭 Данных за {title} нет.", parse_mode="HTML")

    text = f"📈 <b>Сводка по хозяйству ({title}):</b>\n\n"
    for diagnosis, count in stats:
        icon = "🟢" if "Здоровый" in diagnosis else "🔴"
        text += f"{icon} <b>{diagnosis}:</b> {count} шт.\n"

    await callback.message.answer(text, parse_mode="HTML")
    await callback.answer()

@dp.callback_query(F.data.startswith("excel_"))
async def send_excel_report(callback: types.CallbackQuery):
    user_ent = await get_user_enterprise(callback.from_user.id)
    period = callback.data.split("_")[1]

    await callback.message.answer("⏳ <i>Формируем документ...</i>", parse_mode="HTML")
    await callback.answer()

    file_path = await export_to_csv_async(user_ent, period)
    if file_path:
        await callback.message.answer_document(FSInputFile(file_path), caption=f"📁 Выгрузка ({user_ent} - {period})")
    else:
        await callback.message.answer("❌ Нет данных для выгрузки в этом периоде.")

async def handle_media(message: types.Message, file_id: str, with_gps: bool):
    user_ent = await get_user_enterprise(message.from_user.id) or "Без привязки к хозяйству"
    processing_msg = await message.answer("📁 Файл получил! Изучаю снимок и метаданные...")

    try:
        file_info = await bot.get_file(file_id)
        downloaded_file = await bot.download_file(file_info.file_path)

        diagnosis, confidence, coordinates = await asyncio.to_thread(process_image, downloaded_file.read())
        await save_diagnostic(message.from_user.id, user_ent, diagnosis, confidence, coordinates)

        text = f"<b>РЕЗУЛЬТАТ ({user_ent})</b>\n<b>Диагноз:</b> {diagnosis}\n<b>Уверенность:</b> {confidence:.2f}%\n\n"
        if not with_gps:
            text += "📍 <i>Локация недоступна. Отправь снимок как ФАЙЛ.</i>"
        elif coordinates:
            text += f"📍 <b>Локация:</b> <a href='https://www.google.com/maps?q={coordinates[0]},{coordinates[1]}'>Показать на карте</a>"
        else:
            text += "📍 <i>Геометка не найдена в файле.</i>"

        await processing_msg.edit_text(
            text, parse_mode="HTML", reply_markup=get_action_keyboard(diagnosis, coordinates), disable_web_page_preview=True
        )
    except Exception as e:
        logger.error(f"Ошибка при анализе фото: {e}")
        await processing_msg.edit_text("❌ Ошибка при обработке файла.")

@dp.message(F.document)
async def handle_document(message: types.Message):
    if not message.document.mime_type.startswith('image/'):
        return await message.answer("❌ Отправь файл изображения (JPEG).")
    await handle_media(message, message.document.file_id, with_gps=True)

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    await handle_media(message, message.photo[-1].file_id, with_gps=False)

@dp.callback_query(F.data.startswith("point_risk_"))
async def call_point_weather_risk(callback: types.CallbackQuery):
    if "no_geo" in callback.data:
        return await callback.answer("❌ У этого снимка отсутствуют GPS-координаты.", show_alert=True)

    parts = callback.data.split("_")
    lat, lon = float(parts[2]), float(parts[3])

    if not WEATHER_API_KEY:
        return await callback.answer("❌ Не настроен API-ключ OpenWeatherMap.", show_alert=True)

    await callback.message.answer("🛰 Запрос локального прогноза погоды...", parse_mode="HTML")
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric&lang=ru"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    forecast_list = data['list'][:24]
                    max_humidity = max(item['main']['humidity'] for item in forecast_list)
                    avg_temp = sum(item['main']['temp'] for item in forecast_list) / len(forecast_list)

                    report = (
                        f"☁️ <b>ЛОКАЛЬНЫЙ ПРОГНОЗ ПОГОДЫ ДЛЯ ОЧАГА</b>\n"
                        f"📍 Координаты: <code>{lat:.4f}, {lon:.4f}</code>\n"
                        f"🌡 Средняя t° (3 дня): <b>{avg_temp:.1f}°C</b> | Пиковая влажность: <b>{max_humidity}%</b>\n\n"
                    )
                    if max_humidity >= 85 and 15 <= avg_temp <= 22:
                        report += "🚨 <b>ОЦЕНКА УГРОЗЫ: КРИТИЧЕСКИЙ РИСК!</b>\nМетеоусловия идеальны для споруляции. Патоген может разрастись за 48–72 часа."
                    else:
                        report += "🟢 <b>ОЦЕНКА УГРОЗЫ: УМЕРЕННАЯ.</b>\nБлижайший прогноз не благоприятствует резкому всплеску болезни в этой точке."
                    await callback.message.answer(report, parse_mode="HTML")
                else:
                    await callback.message.answer("❌ Ошибка метеосервиса.")
    except aiohttp.ClientError:
        await callback.message.answer("❌ Сбой сети.")
    await callback.answer()

@dp.callback_query(F.data == "send_incident")
async def call_incident(callback: types.CallbackQuery):
    user_ent = await get_user_enterprise(callback.from_user.id)
    if not user_ent: return await callback.answer("❌ Вы не привязаны к предприятию!", show_alert=True)

    target_chat_id = await get_dispatcher_chat(user_ent)
    if not target_chat_id: return await callback.answer("❌ Ситуационный центр не настроен!", show_alert=True)

    msg_text = callback.message.html_text

    if "Фитофтороз" in msg_text:
        protocol = (
            "📋 <b>ПРОТОКОЛ РЕАГИРОВАНИЯ (Фитофтороз):</b>\n"
            "1. <b>Статус:</b> Критический риск эпифитотии.\n"
            "2. <b>Действие:</b> Немедленная локальная обработка очага и буферной зоны (20 м).\n"
            "3. <b>Препараты (Госреестр РБ):</b> Системно-трансламинарные фунгициды (Ридомил Голд, Ревус Топ).\n"
            "4. <b>Норма расхода:</b> Рабочая жидкость 200–400 л/га."
        )
    elif "Альтернариоз" in msg_text:
        protocol = (
            "📋 <b>ПРОТОКОЛ РЕАГИРОВАНИЯ (Альтернариоз):</b>\n"
            "1. <b>Статус:</b> Предупреждение (угроза листовому аппарату).\n"
            "2. <b>Действие:</b> Плановая обработка при достижении порога вредоносности.\n"
            "3. <b>Препараты (Госреестр РБ):</b> Фунгициды класса стробилуринов или триазолов (Скор, Квадрис)."
        )
    else:
        protocol = "🟢 Патогенов не выявлено. Мониторинг штатный."

    alert_text = (
        f"🚨 <b>СИГНАЛ С ПОЛЕЙ: {user_ent}</b> 🚨\n\n"
        f"{msg_text}\n\n"
        f"{protocol}\n\n"
        "🚜 <i>Координаты переданы. Требуется выезд опрыскивателя!</i>"
    )

    try:
        await bot.send_message(chat_id=target_chat_id, text=alert_text, parse_mode="HTML")
        await callback.answer("✅ Данные и протокол переданы диспетчеру!", show_alert=True)
    except Exception as e:
        logger.error(f"Ошибка отправки в группу: {e}")
        await callback.answer("❌ Ошибка связи с группой.", show_alert=True)

@dp.callback_query(F.data.startswith("treat_"))
async def call_treatment(callback: types.CallbackQuery):
    await callback.message.answer(
        "💊 <b>Рекомендации Госреестра СЗР РБ:</b>\n"
        "При подтверждении ИИ-диагноза рекомендуется провести прецизионную локальную обработку "
        "фунгицидами класса стробилуринов строго по зафиксированным координатам.",
        parse_mode="HTML"
    )
    await callback.answer()

async def main():
    await init_db()
    logger.info("Бот запущен и слушает Telegram!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())