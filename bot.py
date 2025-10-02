# bot.py
import io
import os
import gc
import time
import asyncio
import logging
from collections import defaultdict
from PIL import Image, UnidentifiedImageError
from video_utils import load_video_frames
import tempfile

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from model_utils import load_model_and_tokenizer
from analysis_utils import get_detailed_analysis
from image_utils import get_gps_from_exif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Настройки ────────────────────────────────────────────────
MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB
MAX_PIXELS = 25_000_000  # ~25 MP
RATE_LIMIT_SEC = 20  # не чаще одного анализа в 20 сек на пользователя

TOKEN = "8377634357:AAFhH06kmT6_8AKPCadKFlx8Jd63gbK4xcA"
if not TOKEN:
    raise RuntimeError("Установите TELEGRAM_BOT_TOKEN в переменных окружения")

# ── Модель ───────────────────────────────────────────────────
model_cache = {}


def get_model():
    if not model_cache:
        model, tokenizer, device, torch_dtype = load_model_and_tokenizer()
        model_cache.update(
            {
                "model": model,
                "tokenizer": tokenizer,
                "device": device,
                "torch_dtype": torch_dtype,
            }
        )
    return (
        model_cache["model"],
        model_cache["tokenizer"],
        model_cache["device"],
        model_cache["torch_dtype"],
    )


# ── Состояние пользователей ──────────────────────────────────
last_request_time = defaultdict(float)  # user_id -> timestamp
user_busy = set()  # кто сейчас в анализе


async def handle_video(
    update: Update, context: ContextTypes.DEFAULT_TYPE, video_bytes: bytes
):
    chat_id = update.effective_chat.id
    start = time.monotonic()

    progress = await update.message.reply_text(
        "🎬 Выполняю анализ видео…", reply_to_message_id=update.message.message_id
    )
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # Сохраняем видео во временный файл
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        model, tokenizer, device, torch_dtype = get_model()
        try:
            pixel_values, num_patches_list = load_video_frames(
                tmp.name, device=device, dtype=torch_dtype
            )
        except Exception:
            await progress.edit_text("❌ Ошибка при обработке видео. 63")
            return

    # Формируем вопрос для модели
    video_prefix = "".join(
        [f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))]
    )
    question = (
        video_prefix
        + """
Оценка здоровья: Оцени здоровье растения числом от 1 до 10. 
Формат: 'Здоровье: N баллов из 10'.

Вид и порода: Определи вид растения по изображению. 
Верни ровно одно название из списка: 
'Берёза, Клен, Дуб, Кедр, Ива, Тополь, Сосна, Ель, Туя, Рябина, Каштан, Осина, Липа, Ясень, фруктовые растения'. 
Если кустарник – выбери одно название из списка кустарников. 
Формат ответа: 'Вид и порода: <одно название>. Тип: дерево/кустарник/пень.'

Процент сухих ветвей: Оцени процент сухих ветвей и выбери только один вариант: 
0-25%, 25-50%, 50-75%, или 75-100%. 
Формат: 'Сухие ветви: <вариант>'.

Угол наклона: Оцени угол наклона дерева. Формат: 'Угол наклона: N градусов'.

Перечисли видимые дефекты: трещины, дупла, сломанные ветви, "
            "механические повреждения, отслоение коры и т.д. "
            "Если их нет, напиши: 'Дефектов не обнаружено'."
"""
    )

    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: model.chat(
                tokenizer,
                pixel_values,
                question,
                dict(max_new_tokens=350, do_sample=False),
                num_patches_list=num_patches_list,
            ),
        )
    except Exception:
        await progress.edit_text("❌ Ошибка при анализе видео.")
        return

    duration = time.monotonic() - start
    out_text = f"<b>🔍 Детальный анализ видео</b>\n\n{response}\n\nОбработка заняла {duration:.2f} секунд."
    await progress.edit_text(out_text, parse_mode="HTML")


# ── Общие утилиты ────────────────────────────────────────────
async def analyze_one_photo(
    update: Update, context: ContextTypes.DEFAULT_TYPE, img_bytes: bytes
):
    """Анализ одного изображения (байты)."""
    chat_id = update.effective_chat.id
    start = time.monotonic()

    # базовые проверки
    if len(img_bytes) > MAX_IMAGE_BYTES:
        await update.message.reply_text(
            "⚠️ Файл слишком большой (макс 8 MB).",
            reply_to_message_id=update.message.message_id,
        )
        return

    try:
        pil_img = Image.open(io.BytesIO(img_bytes))
        pil_img.verify()
    except UnidentifiedImageError:
        await update.message.reply_text(
            "⚠️ Это не похоже на изображение.",
            reply_to_message_id=update.message.message_id,
        )
        return

    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if pil_img.width * pil_img.height > MAX_PIXELS:
        await update.message.reply_text(
            "⚠️ Слишком высокое разрешение.",
            reply_to_message_id=update.message.message_id,
        )
        return

    gps = None
    try:
        gps = get_gps_from_exif(io.BytesIO(img_bytes))
    except Exception:
        pass

    # индикатор
    progress = await update.message.reply_text(
        "🔎 Выполняю детальный анализ…", reply_to_message_id=update.message.message_id
    )
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    # убираем EXIF
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    img_for_model = Image.open(buf).convert("RGB")

    model, tokenizer, device, torch_dtype = get_model()

    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: get_detailed_analysis(
                img_for_model,
                model,
                tokenizer,
                device,
                torch_dtype,
                tree_description="",
            ),
        )
    except Exception:
        logger.exception("Ошибка анализа")
        await progress.edit_text("❌ Ошибка при анализе. ")
        return

    # Формируем текст отчета
    out_lines = ["<b>🔍 Детальный анализ изображения</b>", ""]

    # Основные результаты
    for k, v in results.items():
        if k == "Главное описание":
            continue
        if k == "Дополнительное описание":
            continue  # отложим до конца
        out_lines.append(f"<b>{k}:</b> {v}")

    # Добавляем GPS, если есть
    if gps:
        lat, lon = gps.get("latitude"), gps.get("longitude")
        if lat and lon:
            out_lines.append("")
            out_lines.append(f"📍 Координаты: {lat:.6f}, {lon:.6f}")
            out_lines.append(f"https://maps.google.com/?q={lat},{lon}")

    # Добавляем Дополнительное описание последним
    extra_desc = results.get("Дополнительное описание", "")
    if extra_desc:
        out_lines.append("")  # пустая строка перед
        out_lines.append(f"<b>Дополнительное описание:</b> {extra_desc}")

    # Добавляем длительность обработки
    duration = time.monotonic() - start
    out_lines.append("")
    out_lines.append(f"Обработка заняла {duration:.2f} секунд.")

    # Отправляем сообщение
    await progress.edit_text("\n".join(out_lines), parse_mode="HTML")


async def handle_images(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработка одиночного фото, альбома или видео.
    Если фото часть альбома, собираем все фото с одинаковым media_group_id.
    """
    user_id = update.effective_user.id
    now = time.monotonic()

    # --- проверка лимита ---
    if now - last_request_time[user_id] < RATE_LIMIT_SEC:
        await update.message.reply_text(
            f"⏳ Слишком часто! Подождите {RATE_LIMIT_SEC} сек между запросами.",
            reply_to_message_id=update.message.message_id,
        )
        return

    # --- проверка занятости ---
    if user_id in user_busy:
        await update.message.reply_text(
            "⚠️ Предыдущий анализ ещё выполняется, дождитесь завершения.",
            reply_to_message_id=update.message.message_id,
        )
        return

    # --- Обработка видео ---
    if update.message.video:
        user_busy.add(user_id)
        last_request_time[user_id] = now
        try:
            if update.message.video.file_size > 20 * 1024 * 1024:  # 20 MB
                await update.message.reply_text(
                    "⚠️ Видео слишком большое (макс. 20 MB). Попробуйте обрезать или сжать его.",
                    reply_to_message_id=update.message.message_id,
                )
                return
            file = await update.message.video.get_file()
            video_bytes = await file.download_as_bytearray()
            await handle_video(update, context, bytes(video_bytes))
        finally:
            user_busy.discard(user_id)
        return

    # === Сбор фото альбома ===
    group_id = update.message.media_group_id
    if group_id:
        context.chat_data.setdefault(group_id, []).append(update.message)

        await asyncio.sleep(1.0)  # небольшая задержка, чтобы собрать все фото
        if group_id not in context.chat_data:
            return  # уже обработали

        msgs = sorted(context.chat_data.pop(group_id), key=lambda m: m.date)
        user_busy.add(user_id)
        last_request_time[user_id] = now
        try:
            for m in msgs:
                file = await m.photo[-1].get_file()
                img_bytes = await file.download_as_bytearray()
                await analyze_one_photo(m, context, bytes(img_bytes))
        finally:
            user_busy.discard(user_id)
        return

    # === Одиночное фото ===
    user_busy.add(user_id)
    last_request_time[user_id] = now
    try:
        file = await update.message.photo[-1].get_file()
        img_bytes = await file.download_as_bytearray()
        await analyze_one_photo(update, context, bytes(img_bytes))
    finally:
        user_busy.discard(user_id)


async def album_collector(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Собираем фото альбома, чтобы потом отдать все разом."""
    if not context.chat_data:
        context.chat_data.clear()
    gid = update.message.media_group_id
    if gid:
        context.chat_data.setdefault(gid, []).append(update.message)
    # реальный анализ произойдет в handle_images (оно же вызывается для каждого фото в группе)


# Если не фото
async def handle_not_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Я жду только изображение 📷", reply_to_message_id=update.message.message_id
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Пришли фото (или альбом), и я выполню детальный анализ.",
        reply_to_message_id=update.message.message_id,
    )


async def clear_cache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model_cache.clear()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    await update.message.reply_text(
        "Кэш модели очищен.", reply_to_message_id=update.message.message_id
    )


# ── Точка входа ──────────────────────────────────────────────
def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear_cache", clear_cache))

    # одно фото И альбомы — всё ловит этот один хэндлер
    app.add_handler(MessageHandler(filters.PHOTO, handle_images))

    # видео
    app.add_handler(MessageHandler(filters.VIDEO, handle_images))

    # всё остальное
    app.add_handler(
        MessageHandler(
            ~filters.PHOTO & ~filters.VIDEO & ~filters.COMMAND, handle_not_photo
        )
    )

    app.run_polling()


if __name__ == "__main__":
    main()
