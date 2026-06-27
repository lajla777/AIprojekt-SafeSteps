import time
from nicegui import run, ui
from config import config
from serial_live import set_serial_live_enabled
from state import state
from ui_common import hover_tts, page_shell, render_log


@ui.page('/live-tof')
def live_tof_page() -> None:
    page_shell('Live ToF STM32')

    with ui.column().classes('w-full p-5 gap-4'):
        ui.label('Live zajem ToF podatkov prek STM32').classes('page-title')

        with ui.row().classes('w-full gap-3'):
            with ui.column().classes('metric-card flex-1'):
                ui.label('Povezava').classes('text-xs text-gray-400 uppercase tracking-wide')
                conn = ui.label('-').classes('metric-val')
                endpoint = ui.label('').classes('text-sm text-gray-500')
                search_button = ui.button('Začni live zajem', icon='play_arrow').props('color=primary')
                hover_tts(search_button, 'Začni ali ustavi zajem ToF podatkov.')

            with ui.column().classes('metric-card flex-1'):
                ui.label('Live zajemi').classes('text-xs text-gray-400 uppercase tracking-wide')
                captures = ui.label('0').classes('metric-val')
                last_seen = ui.label('-').classes('text-sm text-gray-500')

            with ui.column().classes('metric-card flex-1'):
                ui.label('Napoved ToF mreže').classes('text-xs text-gray-400 uppercase tracking-wide')
                prediction = ui.label('-').classes('metric-val')
                prediction_meta = ui.label('-').classes('text-sm text-gray-500 break-all')

        with ui.column().classes('metric-card w-full'):
            ui.label('Komunikacijski log').classes('text-xs text-gray-400 uppercase tracking-wide mb-2')
            log_box = ui.column().classes('w-full')

        async def toggle_tof_search() -> None:
            search_button.disable()
            try:
                await run.io_bound(set_serial_live_enabled, not state.tof_search_enabled)
            finally:
                search_button.enable()
                search_button.set_text('Ustavi live zajem' if state.tof_search_enabled else 'Začni live zajem')
                search_button.props('icon=stop' if state.tof_search_enabled else 'icon=play_arrow')

        search_button.on_click(toggle_tof_search)

        def refresh() -> None:
            if state.stm_connected:
                conn_text = 'live'
                conn_class = 'text-green-600'
            elif state.tof_search_enabled:
                conn_text = 'capturing'
                conn_class = 'text-amber-500'
            else:
                conn_text = 'stopped'
                conn_class = 'text-gray-400'

            conn.set_text(conn_text)
            conn.classes(
                remove='text-green-600 text-red-500 text-amber-500 text-gray-400',
                add=conn_class,
            )

            search_button.set_text('Ustavi live zajem' if state.tof_search_enabled else 'Začni live zajem')
            search_button.props('icon=stop' if state.tof_search_enabled else 'icon=play_arrow')

            endpoint.set_text(
                f'Serial {state.tof_live_port or config.stm32_serial_port or "auto"} @ {config.stm32_baudrate}'
            )
            captures.set_text(str(state.tof_packets))
            last_seen.set_text(
                f'Zadnji zajem pred {int(time.time() - state.tof_last_seen)} s'
                if state.tof_last_seen else 'Ni live zajema'
            )

            if state.tof_prediction_text:
                prediction.set_text(f'{state.tof_prediction_text} ({int(state.tof_confidence * 100)}%)')
                prediction_meta.set_text(f'{state.tof_inference_ms:.0f} ms, {state.tof_last_raw}')
            else:
                prediction.set_text('-')
                prediction_meta.set_text(state.tof_last_raw or '-')

            log_box.clear()
            with log_box:
                render_log(16)

        ui.timer(1.0, refresh)
        refresh()
