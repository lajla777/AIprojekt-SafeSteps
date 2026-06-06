import base64
from nicegui import ui
from camera import set_camera_enabled
from config import config
from image_utils import draw_detection_boxes, image_bytes_to_array, read_upload_event, results_to_detections
from src.models.image.predict import predict_source
from state import add_log, state
from tts import speak
from ui_common import hover_tts, page_shell
from pathlib import Path


@ui.page('/image-network')
def image_network_page() -> None:
    page_shell('Mreža slik')

    with ui.column().classes('w-full p-5 gap-4'):
        ui.label('Nevronska mreža za slike').classes('page-title')

        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('metric-card flex-1'):
                ui.label('Zaznavanje iz slike').classes('text-sm text-gray-400 uppercase tracking-wide')
                ui.label('Klikni na okno in izberi sliko.').classes('text-sm text-gray-500')
                upload = ui.upload(label='Izberi sliko', auto_upload=True).props('accept=image/*').classes('hidden')
                upload_card = ui.card().classes(
                    'w-full p-0 overflow-hidden cursor-pointer border border-gray-200 bg-gray-50'
                ).on('click', lambda: upload.run_method('pickFiles'))
                hover_tts(upload_card, 'Klikni za izbiro slike za zaznavanje objektov.')
                with upload_card:
                    uploaded_image = ui.image().classes('w-full object-contain')
                    upload_hint = ui.label('Klikni za izbiro slike.').classes('absolute-center text-gray-500 text-lg').on('click', lambda: upload.run_method('pickFiles'))
                    

            with ui.column().classes('flex-1 gap-3'):
                with ui.column().classes('metric-card w-full'):
                    ui.label('Model').classes('text-sm text-gray-400 uppercase tracking-wide')
                    ui.label(Path(config.yolo_model).name).classes('metric-val text-lg')
                    conf_label = ui.label('').classes('text-sm text-gray-500')

                with ui.column().classes('metric-card w-full'):
                    ui.label('Rezultat naložene slike').classes('text-sm text-gray-400 uppercase tracking-wide mb-2')
                    upload_inference = ui.label('- ms').classes('metric-val text-lg')
                    uploaded_results = ui.column().classes('w-full gap-2')

        with ui.column().classes('metric-card w-full gap-3'):
            ui.label('Kamera').classes('text-sm text-gray-400 uppercase tracking-wide')
            with ui.row().classes('w-full gap-3'):
                with ui.column().classes('flex-1'):
                    ui.label('Status kamere').classes('text-sm text-gray-400 uppercase tracking-wide')
                    cam_status = ui.label('-').classes('metric-val')
                    camera_hint = ui.label(f'Camera index: {config.camera_index}').classes('text-sm text-gray-500')
                    camera_button = ui.button('Vklopi kamero', icon='videocam')
                    hover_tts(camera_button, 'Vklopi ali izklopi kamero.')

                with ui.column().classes('flex-1'):
                    ui.label('Čas prepoznave objekta').classes('text-sm text-gray-400 uppercase tracking-wide')
                    inference = ui.label('- ms').classes('metric-val')

            ui.separator()
            ui.label('Zaznani objekti kamere').classes('text-sm text-gray-400 uppercase tracking-wide')
            detections_box = ui.column().classes('w-full gap-2')

        def render_uploaded_results() -> None:
            upload_inference.set_text(
                f'{state.uploaded_inference_ms:.0f} ms' if state.uploaded_inference_ms > 0 else '- ms'
            )
            uploaded_results.clear()

            with uploaded_results:
                if not state.uploaded_detections:
                    ui.label('Ni rezultatov.').classes('text-sm text-gray-500')

                for item in state.uploaded_detections:
                    with ui.row().classes('items-center justify-between w-full border-b border-gray-100 py-2'):
                        ui.label(item.get('text', item['label'])).classes('text-base font-medium')
                        ui.label(f"{int(item['conf'] * 100)}%").classes('text-sm text-gray-500')

        async def detect_uploaded_image(event) -> None:
            try:
                image_bytes, _, mime_type = await read_upload_event(event)

                state.uploaded_image_data_url = (
                    f'data:{mime_type};base64,' + base64.b64encode(image_bytes).decode('ascii')
                )
                uploaded_image.source = state.uploaded_image_data_url
                upload_hint.visible = False

                source = image_bytes_to_array(image_bytes)
                results, names, state.uploaded_inference_ms = predict_source(
                    source,
                    conf=config.yolo_confidence,
                )
                state.uploaded_detections = results_to_detections(results, names, config.yolo_confidence)
                if state.uploaded_detections:
                    uploaded_image.source = draw_detection_boxes(image_bytes, state.uploaded_detections)
                add_log(f'Uploaded image detected: {len(state.uploaded_detections)} objects', 'ok')

                if state.uploaded_detections:
                    labels = ', '.join(item.get('text', item['label']) for item in state.uploaded_detections[:3])
                    speak(f'Na sliki zaznam: {labels}.')
                else:
                    speak('Na sliki ni zaznanih objektov.')

                render_uploaded_results()

            except Exception as exc:
                add_log(f'Uploaded image detection failed: {exc}', 'danger')
                ui.notify(f'Zaznavanje slike ni uspelo: {exc}', type='negative')

        upload.on_upload(detect_uploaded_image)

        def toggle_camera() -> None:
            set_camera_enabled(not state.camera_enabled)
            camera_button.set_text('Izklopi kamero' if state.camera_enabled else 'Vklopi kamero')
            camera_button.props('icon=videocam_off' if state.camera_enabled else 'icon=videocam')

        camera_button.on_click(toggle_camera)

        def refresh() -> None:
            if state.cam_connected:
                status_text = 'active'
                status_class = 'text-green-600'
            elif state.camera_enabled:
                status_text = 'starting'
                status_class = 'text-amber-500'
            else:
                status_text = 'off'
                status_class = 'text-gray-400'

            cam_status.set_text(status_text)
            cam_status.classes(
                remove='text-green-600 text-red-500 text-amber-500 text-gray-400',
                add=status_class,
            )
            camera_hint.set_text(f'Camera index: {config.camera_index}')
            camera_button.set_text('Izklopi kamero' if state.camera_enabled else 'Vklopi kamero')
            camera_button.props('icon=videocam_off' if state.camera_enabled else 'icon=videocam')
            conf_label.set_text(f'Confidence threshold: {config.yolo_confidence:.2f}')
            inference.set_text(f'{state.inference_ms:.0f} ms' if state.inference_ms > 0 else '- ms')

            detections_box.clear()
            with detections_box:
                if not state.detections:
                    ui.label('Ni zaznav.').classes('text-sm text-gray-500')

                for item in state.detections:
                    with ui.row().classes('items-center justify-between w-full border-b border-gray-100 py-2'):
                        ui.label(item.get('text', item['label'])).classes('text-base font-medium')
                        ui.label(f"{int(item['conf'] * 100)}%").classes('text-sm text-gray-500')

        ui.timer(0.8, refresh)
        refresh()
        render_uploaded_results()
