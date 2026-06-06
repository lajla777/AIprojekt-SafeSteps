import base64
import inspect
import io


IMAGE_LABEL_TEXT = {
    'person': 'oseba',
    'car': 'avtomobil',
}

def image_label_text(label: str) -> str:
    return IMAGE_LABEL_TEXT.get(label, label.replace('_', ' '))


async def read_upload_event(event) -> tuple[bytes, str, str]:
    uploaded_file = getattr(event, 'content', None) or getattr(event, 'file', None)
    file_obj = uploaded_file

    if hasattr(file_obj, 'file'):
        file_obj = file_obj.file

    if file_obj is None:
        raise ValueError('Upload event does not contain a readable file')

    if hasattr(file_obj, 'seek'):
        file_obj.seek(0)

    file_bytes = file_obj.read()

    if inspect.isawaitable(file_bytes):
        file_bytes = await file_bytes

    if isinstance(file_bytes, str):
        file_bytes = file_bytes.encode('utf-8')

    if not file_bytes:
        raise ValueError('Uploaded file is empty')

    name = (
        getattr(event, 'name', '')
        or getattr(event, 'filename', '')
        or getattr(uploaded_file, 'filename', '')
        or getattr(uploaded_file, 'name', '')
        or getattr(file_obj, 'filename', '')
        or getattr(file_obj, 'name', '')
        or 'uploaded'
    )
    mime_type = (
        getattr(event, 'type', '')
        or getattr(event, 'content_type', '')
        or getattr(uploaded_file, 'content_type', '')
        or 'application/octet-stream'
    )
    suffix = '.' + name.rsplit('.', 1)[-1].lower() if '.' in name else ''

    return file_bytes, suffix, mime_type


def image_bytes_to_array(image_bytes: bytes):
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return np.array(image)


def results_to_detections(results, names: dict, conf_threshold: float) -> list[dict]:
    detections = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id = int(box.cls[0])
            label = names[cls_id]
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            detections.append({
                'label': label,
                'text': image_label_text(label),
                'conf': round(conf, 2),
                'bbox': [x1, y1, x2, y2],
            })

    return detections


def draw_detection_boxes(image_bytes: bytes, detections: list[dict]) -> str:
    from state import add_log

    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        palette = ['#2f7d13', '#c77700', '#0f766e', '#b91c1c', '#2563eb']
        for index, item in enumerate(detections):
            bbox = item.get('bbox')
            if not bbox or len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            color = palette[index % len(palette)]
            text = f"{item.get('text', item['label'])} {int(item['conf'] * 100)}%"

            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_width = right - left
            text_height = bottom - top
            label_y = max(0, y1 - text_height - 8)

            draw.rectangle(
                (x1, label_y, x1 + text_width + 10, label_y + text_height + 6),
                fill=color,
            )
            draw.text((x1 + 5, label_y + 3), text, fill='white', font=font)

        output = io.BytesIO()
        image.save(output, format='JPEG', quality=92)
        encoded = base64.b64encode(output.getvalue()).decode('ascii')
        return f'data:image/jpeg;base64,{encoded}'
    except Exception as exc:
        add_log(f'Drawing detection boxes failed: {exc}', 'warn')
        return 'data:image/jpeg;base64,' + base64.b64encode(image_bytes).decode('ascii')
