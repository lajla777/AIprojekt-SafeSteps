from pathlib import Path

from nicegui import ui

from config import config
from state import add_log
from tts import speak
from ui_common import hover_tts, page_shell


@ui.page('/settings')
def settings_page() -> None:
    page_shell('Nastavitve')

    with ui.column().classes('w-full p-6 gap-5 max-w-[1480px] mx-auto'):
        with ui.row().classes(
            'w-full items-center justify-between gap-4 bg-white border border-gray-200 rounded-lg px-6 py-5'
        ):
            with ui.column().classes('gap-1'):
                ui.label('Nastavitve sistema').classes('page-title')
                ui.label('Povezava, modeli, live ToF obdelava in govorna opozorila.').classes('settings-help')

            save = ui.button('Shrani nastavitve', icon='save').props('color=primary').classes(
                'px-8 py-3 text-lg min-w-64'
            )
            hover_tts(save, 'Shrani nastavitve aplikacije.')

        with ui.row().classes('w-full gap-5 items-start'):
            with ui.column().classes('w-[430px] gap-5 shrink-0'):
                with ui.column().classes('metric-card w-full gap-4'):
                    ui.label('STM32 povezava').classes('settings-title')
                    ui.label('Live zajem prek serial COM porta.').classes('settings-help')

                    with ui.grid(columns=2).classes('w-full gap-x-4 gap-y-4 settings-controls'):
                        with ui.column().classes('w-full gap-1'):
                            ui.label('Serial port').classes('settings-field-label')
                            serial_port_input = ui.input(
                                value=config.stm32_serial_port,
                                placeholder='auto ali COM5',
                            ).props('dense').classes('w-full')
                        with ui.column().classes('w-full gap-1'):
                            ui.label('Baudrate').classes('settings-field-label')
                            baudrate_input = ui.number(
                                value=config.stm32_baudrate,
                                min=9600,
                                max=921600,
                            ).props('dense').classes('w-full')
                        with ui.column().classes('w-full gap-1'):
                            ui.label('VID').classes('settings-field-label')
                            vid_input = ui.input(value=config.stm32_vid).props('dense').classes('w-full')
                        with ui.column().classes('w-full gap-1'):
                            ui.label('PID').classes('settings-field-label')
                            pid_input = ui.input(value=config.stm32_pid).props('dense').classes('w-full')

                with ui.column().classes('metric-card w-full gap-4'):
                    ui.label('TTS govor').classes('settings-title')
                    ui.label('Piper govor za hover gumbe, rezultate in opozorila.').classes('settings-help')
                    tts_enabled = ui.switch('TTS enabled', value=config.tts_enabled).classes('settings-controls')
                    with ui.column().classes('w-full gap-1'):
                        ui.label('Piper model').classes('settings-field-label')
                        tts_model_input = ui.input(value=config.tts_model).props('dense').classes('w-full')
                    ui.label(Path(config.tts_model).name).classes('settings-help truncate')

            with ui.column().classes('flex-1 min-w-[760px] gap-5'):
                with ui.column().classes('metric-card w-full gap-4'):
                    ui.label('Modeli').classes('settings-title')
                    ui.label('Poti se samodejno najdejo iz projekta, ročno jih spremeni samo pri testiranju.').classes(
                        'settings-help'
                    )

                    with ui.grid(columns=2).classes('w-full gap-5 settings-controls'):
                        with ui.column().classes('w-full gap-2'):
                            ui.label('Mreža slik').classes('settings-title text-base')
                            ui.label('YOLO model').classes('settings-field-label')
                            model_input = ui.input(value=config.yolo_model).props('dense').classes('w-full')
                            ui.label(Path(config.yolo_model).name).classes('settings-help truncate')
                            conf_slider = ui.slider(
                                min=0.1,
                                max=0.95,
                                step=0.05,
                                value=config.yolo_confidence,
                            ).classes('w-full')
                            conf_label = ui.label('')

                        with ui.column().classes('w-full gap-2'):
                            ui.label('Mreža ToF').classes('settings-title text-base')
                            ui.label('ToF model').classes('settings-field-label')
                            tof_model_input = ui.input(value=config.tof_model).props('dense').classes('w-full')
                            ui.label(Path(config.tof_model).name).classes('settings-help truncate')

                with ui.column().classes('metric-card w-full gap-4'):
                    ui.label('Live ToF obdelava').classes('settings-title')
                    ui.label('Te nastavitve vplivajo na stabilnost live napovedi iz sweepov.').classes(
                        'settings-help'
                    )

                    with ui.grid(columns=2).classes('w-full gap-6 settings-controls'):
                        with ui.column().classes('w-full gap-3'):
                            live_valid_label = ui.label('')
                            live_valid_slider = ui.slider(
                                min=0.05,
                                max=0.60,
                                step=0.05,
                                value=config.tof_no_obstacle_valid_ratio,
                            ).classes('w-full')
                            obstacle_conf_label = ui.label('')
                            obstacle_conf_slider = ui.slider(
                                min=0.30,
                                max=0.90,
                                step=0.05,
                                value=config.tof_min_obstacle_confidence,
                            ).classes('w-full')
                            no_obstacle_dist_label = ui.label('')
                            no_obstacle_dist_slider = ui.slider(
                                min=500,
                                max=3500,
                                step=100,
                                value=config.tof_no_obstacle_distance_mm,
                            ).classes('w-full')

                        with ui.column().classes('w-full gap-3'):
                            mirror_lr = ui.switch(
                                'Zrcali levo/desno za live ToF',
                                value=config.tof_mirror_left_right,
                            )
                            geometry_fallback = ui.switch(
                                'Uporabi geometrijo pri negotovi mreži',
                                value=config.tof_use_geometry_fallback,
                            )
                            center_angle_label = ui.label('')
                            center_angle_slider = ui.slider(
                                min=3,
                                max=25,
                                step=1,
                                value=config.tof_center_angle_deg,
                            ).classes('w-full')

                with ui.column().classes('metric-card w-full gap-4'):
                    ui.label('Opozorila za bližino').classes('settings-title')
                    ui.label(
                        'Pragovi se uporabijo po live ToF napovedi: danger sproži nujno opozorilo, '
                        'warning pa mehkejše TTS opozorilo.'
                    ).classes('settings-help')

                    with ui.grid(columns=2).classes('w-full gap-6 settings-controls'):
                        with ui.column().classes('w-full gap-2'):
                            ui.label('Danger prag').classes('settings-field-label')
                            danger_slider = ui.slider(
                                min=0.1,
                                max=1.0,
                                step=0.05,
                                value=config.danger_dist,
                            ).classes('w-full')
                            danger_label = ui.label('')

                        with ui.column().classes('w-full gap-2'):
                            ui.label('Warning prag').classes('settings-field-label')
                            warn_slider = ui.slider(
                                min=0.2,
                                max=2.0,
                                step=0.05,
                                value=config.warn_dist,
                            ).classes('w-full')
                            warn_label = ui.label('')

        def update_labels() -> None:
            conf_label.set_text(f'Confidence: {float(conf_slider.value):.2f}')
            live_valid_label.set_text(
                f'Ni ovire, če je veljavnih razdalj manj kot: {float(live_valid_slider.value):.0%}'
            )
            obstacle_conf_label.set_text(
                f'Minimalno zaupanje za oviro: {float(obstacle_conf_slider.value):.2f}'
            )
            no_obstacle_dist_label.set_text(
                f'Brez bližnje ovire nad: {float(no_obstacle_dist_slider.value):.0f} mm'
            )
            center_angle_label.set_text(
                f'Kot za oviro spredaj: +/- {float(center_angle_slider.value):.0f} stopinj'
            )
            danger_label.set_text(f'Danger: {float(danger_slider.value):.2f} m')
            warn_label.set_text(f'Warning: {float(warn_slider.value):.2f} m')

        conf_slider.on_value_change(lambda _: update_labels())
        live_valid_slider.on_value_change(lambda _: update_labels())
        obstacle_conf_slider.on_value_change(lambda _: update_labels())
        no_obstacle_dist_slider.on_value_change(lambda _: update_labels())
        center_angle_slider.on_value_change(lambda _: update_labels())
        danger_slider.on_value_change(lambda _: update_labels())
        warn_slider.on_value_change(lambda _: update_labels())
        update_labels()

        def save_settings() -> None:
            config.stm32_serial_port = str(serial_port_input.value or '').strip()
            config.stm32_baudrate = int(baudrate_input.value)
            config.stm32_vid = str(vid_input.value).strip()
            config.stm32_pid = str(pid_input.value).strip()
            config.yolo_model = str(model_input.value).strip()
            config.yolo_confidence = float(conf_slider.value)
            config.tof_model = str(tof_model_input.value).strip()
            config.tof_no_obstacle_valid_ratio = float(live_valid_slider.value)
            config.tof_min_obstacle_confidence = float(obstacle_conf_slider.value)
            config.tof_no_obstacle_distance_mm = float(no_obstacle_dist_slider.value)
            config.tof_mirror_left_right = bool(mirror_lr.value)
            config.tof_use_geometry_fallback = bool(geometry_fallback.value)
            config.tof_center_angle_deg = float(center_angle_slider.value)
            config.danger_dist = float(danger_slider.value)
            config.warn_dist = float(warn_slider.value)
            config.tts_enabled = bool(tts_enabled.value)
            config.tts_model = str(tts_model_input.value).strip()

            add_log('Settings saved', 'ok')
            speak('Nastavitve so shranjene.')
            ui.notify('Nastavitve shranjene.', type='positive')

        save.on_click(save_settings)
