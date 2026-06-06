import asyncio
from nicegui import ui
from config import config
from state import state
from tts import speak

def dist_color(distance: float) -> str:
    if distance < 0:
        return 'text-gray-400'
    if distance < config.danger_dist:
        return 'text-red-500'
    if distance < config.warn_dist:
        return 'text-amber-500'
    return 'text-green-600'


def dist_zone(distance: float) -> tuple[str, str]:
    if distance < 0:
        return 'no data', 'bg-gray-100 text-gray-500'
    if distance < config.danger_dist:
        return 'DANGER', 'bg-red-100 text-red-700'
    if distance < config.warn_dist:
        return 'WARNING', 'bg-amber-100 text-amber-700'
    return 'SAFE', 'bg-green-100 text-green-700'


def status_dot(connected: bool) -> str:
    return 'text-green-600' if connected else 'text-red-400'


def level_class(level: str) -> str:
    return {
        'ok': 'text-green-700',
        'warn': 'text-amber-700',
        'danger': 'text-red-700',
        'info': 'text-gray-600',
    }.get(level, 'text-gray-600')


def hover_tts(element: ui.element, text: str, delay: float = 2.0) -> ui.element:
    hover_state = {'active': False, 'task': None}

    async def delayed_speak() -> None:
        await asyncio.sleep(delay)
        if hover_state['active']:
            speak(text, min_interval=delay)

    def start_hover() -> None:
        hover_state['active'] = True
        task = hover_state.get('task')
        if task and not task.done():
            task.cancel()
        hover_state['task'] = asyncio.create_task(delayed_speak())

    def stop_hover() -> None:
        hover_state['active'] = False
        task = hover_state.get('task')
        if task and not task.done():
            task.cancel()

    element.on('mouseenter', start_hover)
    element.on('mouseleave', stop_hover)
    element.on('click', stop_hover)
    return element


def add_styles() -> None:
    ui.add_head_html(
        """
        <style>
            body {
                background: #f7f7f4;
                font-family: Inter, Arial, sans-serif;
                font-size: 18px;
            }
            .metric-card {
                background: white;
                border-radius: 10px;
                border: 1px solid #e5e5df;
                padding: 24px;
            }
            .metric-card .q-field,
            .metric-card .q-slider,
            .metric-card .q-toggle,
            .metric-card .q-select {
                font-size: 18px;
            }
            .metric-card .q-field__label,
            .metric-card .q-field__native,
            .metric-card .q-field__input,
            .metric-card .q-item__label {
                font-size: 18px;
            }
            .metric-card .q-slider {
                min-height: 38px;
            }
            .home-card {
                background: white;
                border-radius: 10px;
                border: 1px solid #e5e5df;
                padding: 28px;
                min-height: 245px;
                cursor: pointer;
                transition: border-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
            }
            .home-card:hover {
                border-color: #6b006b;
                box-shadow: 0 8px 22px rgba(15, 118, 110, 0.12);
                transform: translateY(-1px);
            }
            .home-step {
                background: white;
                border-radius: 10px;
                border: 1px solid #e5e5df;
                padding: 22px;
                min-height: 120px;
            }
            .metric-val { font-size: 30px; font-weight: 600; line-height: 1.15; }
            .page-title { font-size: 30px; font-weight: 700; }
            .settings-title { font-size: 20px; font-weight: 700; }
            .settings-help { font-size: 16px; color: #5f6b7a; }
            .settings-controls {
                margin-top: 14px;
            }
            .settings-field-label {
                font-size: 16px;
                color: #5f6b7a;
                margin-bottom: 0;
            }
            .sidebar-button {
                min-height: 58px;
                font-size: 18px;
                font-weight: 700;
                padding: 10px 16px;
            }
            .sidebar-button .q-icon {
                font-size: 28px;
                margin-right: 10px;
            }
            .pulse { animation: pulse 1.5s infinite; }
            @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.45} }
        </style>
        """
    )


def render_log(max_items: int = 18) -> None:
    with ui.column().classes('w-full gap-1 max-h-96 overflow-y-auto'):
        for entry in state.log_entries[-max_items:]:
            ui.label(f"[{entry['time']}] {entry['msg']}").classes(
                f"text-xs font-mono {level_class(entry['level'])}"
            )


def page_shell(title: str, show_sidebar: bool = True) -> None:
    add_styles()

    with ui.header().classes('items-center justify-between bg-white text-gray-900 border-b border-gray-200 h-20 px-5'):
        with ui.row().classes('items-center gap-3 cursor-pointer').on('click', lambda: ui.navigate.to('/')) as brand:
            ui.icon('blind', size='35px').style('color: #c8008c')
            #ui.icon('star_rate', size='35px').style('color: #c8008c')
            ui.label('SafeSteps').classes('text-2xl font-semibold')
        hover_tts(brand, 'Odpri začetno stran.')

        with ui.row().classes('items-center gap-4'):
            stm = ui.label('STM32').classes(f'text-sm {status_dot(state.stm_connected)}')
            cam = ui.label('Camera').classes(f'text-sm {status_dot(state.cam_connected)}')
            tts = ui.label('TTS').classes(f'text-sm {status_dot(config.tts_enabled)}')

            def refresh_header() -> None:
                stm.classes(remove='text-green-600 text-red-400', add=status_dot(state.stm_connected))
                cam.classes(remove='text-green-600 text-red-400', add=status_dot(state.cam_connected))
                tts.classes(remove='text-green-600 text-red-400', add=status_dot(config.tts_enabled))

            ui.timer(1.0, refresh_header)

    if show_sidebar:
        with ui.left_drawer(value=True).classes('bg-[#40002f] text-white'):
            b = ui.button('Mreža slik', icon='photo_camera', on_click=lambda: ui.navigate.to('/image-network'))
            b.props('flat align=left').classes('sidebar-button w-full justify-start text-white')
            hover_tts(b, 'Odpri stran za nevronsko mrežo slik.')

            b = ui.button('Mreža ToF', icon='grid_view', on_click=lambda: ui.navigate.to('/tof-network'))
            b.props('flat align=left').classes('sidebar-button w-full justify-start text-white')
            hover_tts(b, 'Odpri stran za ToF mrežo.')

            b = ui.button('Live ToF STM', icon='settings_input_antenna', on_click=lambda: ui.navigate.to('/live-tof'))
            b.props('flat align=left').classes('sidebar-button w-full justify-start text-white')
            hover_tts(b, 'Odpri komunikacijo s STM napravo.')

            b = ui.button('Nastavitve', icon='settings', on_click=lambda: ui.navigate.to('/settings'))
            b.props('flat align=left').classes('sidebar-button w-full justify-start text-white')
            hover_tts(b, 'Odpri nastavitve aplikacije.')
