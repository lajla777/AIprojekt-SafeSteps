import os
import tempfile
from pathlib import Path
from nicegui import ui
from config import config
from image_utils import read_upload_event
from state import state
from tof_model import predict_bin_file
from ui_common import page_shell


@ui.page('/tof-network')
def tof_network_page() -> None:
    page_shell('Mreža ToF')

    with ui.column().classes('w-full p-5 gap-4'):
        ui.label('Nevronska mreža ToF').classes('page-title')

        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('metric-card flex-1'):
                ui.label('Analiza .BIN datoteke').classes('text-xs text-gray-400 uppercase tracking-wide')
                ui.label('Naloži ToF .BIN datoteko in mreža napove lego ovire.').classes('text-sm text-gray-500')
                upload = ui.upload(label='Izberi .BIN datoteko', auto_upload=True).props('accept=.bin,.BIN').classes('w-full')

            with ui.column().classes('metric-card flex-1'):
                ui.label('ToF model').classes('text-xs text-gray-400 uppercase tracking-wide')
                tof_model_path = Path(config.tof_model)
                ui.label(tof_model_path.name).classes('metric-val text-lg')
                ui.label(str(tof_model_path.parent)).classes('text-sm text-gray-500 break-all')

        with ui.column().classes('metric-card w-full'):
            ui.label('Rezultat datoteke').classes('text-xs text-gray-400 uppercase tracking-wide')
            file_prediction = ui.label('-').classes('metric-val text-lg')
            file_meta = ui.label('Naloži .BIN datoteko za prikaz rezultata.').classes('text-sm text-gray-500')
            file_windows = ui.column().classes('w-full gap-1')

        async def analyze_uploaded_tof(event) -> None:
            temp_path = ''
            try:
                file_bytes, suffix, _ = await read_upload_event(event)
                if suffix and suffix.lower() != '.bin':
                    raise ValueError('Naložena datoteka mora biti .BIN.')

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or '.bin') as temp_file:
                    temp_file.write(file_bytes)
                    temp_path = temp_file.name

                result = predict_bin_file(temp_path)
                state.tof_file_result = result

                file_prediction.set_text(f"{result['text']} ({int(result['confidence'] * 100)}%)")
                file_meta.set_text(
                    f"{result['sample_count']} vzorcev, {result['window_count']} oken, "
                    f"{result['tof_fvz']:.1f} Hz, {result['inference_ms']:.0f} ms"
                )
                file_windows.clear()
                with file_windows:
                    for index, window in enumerate(result['windows'][:8], start=1):
                        with ui.row().classes('items-center justify-between w-full border-b border-gray-100 py-1'):
                            ui.label(f"Okno {index}: {window['text']}").classes('text-sm')
                            ui.label(f"{int(window['confidence'] * 100)}%").classes('text-sm text-gray-500')

                ui.notify('ToF datoteka analizirana.', type='positive')
            except Exception as exc:
                ui.notify(f'Analiza ToF datoteke ni uspela: {exc}', type='negative')
                file_prediction.set_text('-')
                file_meta.set_text(str(exc))
                file_windows.clear()
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        upload.on_upload(analyze_uploaded_tof)
