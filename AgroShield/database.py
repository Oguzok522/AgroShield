import sqlite3
import aiosqlite
import csv
import re
import logging
from datetime import datetime, timedelta
import folium
from folium.plugins import HeatMap

logger = logging.getLogger(__name__)
DB_FILE = 'agro_final.sqlite'

def clean_name(text):
    return re.sub(r'[^\w\-]', '_', text)

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

async def calculate_enterprise_center(enterprise_name):
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute(
            "SELECT latitude, longitude FROM diagnostics WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND enterprise = ?",
            (enterprise_name,)
        ) as cursor:
            rows = await cursor.fetchall()

    if not rows: return None
    return sum(r[0] for r in rows) / len(rows), sum(r[1] for r in rows) / len(rows)

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