from nicegui import ui

from ui_common import hover_tts, page_shell


def navigation_card(title: str, text: str, icon: str, target: str, tts_text: str) -> None:
    with ui.column().classes('home-card justify-between gap-5').on('click', lambda: ui.navigate.to(target)) as card:
        with ui.column().classes('gap-4'):
            ui.icon(icon, size='46px').style('color: #c8008c')
            ui.label(title).classes('text-2xl font-semibold')
            ui.label(text).classes('text-base text-gray-500 leading-relaxed')
        ui.icon('arrow_forward', size='28px').classes('text-gray-400 self-end')
    hover_tts(card, tts_text)


@ui.page('/')
def dashboard_page() -> None:
    page_shell('Domov', show_sidebar=False)

    with ui.column().classes('w-full p-8 gap-7'):
        with ui.column().classes('gap-2'):
            ui.label('SafeSteps').classes('text-4xl font-semibold')
            ui.label('Povezava slikovne mreže, ToF mreže, STM komunikacije in TTS podpore.').classes(
                'text-base text-gray-500'
            )

        with ui.grid(columns=4).classes('w-full gap-5'):
            navigation_card(
                'Mreža slik',
                'Naloži sliko ali vklopi kamero za zaznavo objektov.',
                'photo_camera',
                '/image-network',
                'Odpri stran za nevronsko mrežo slik.',
            )
            navigation_card(
                'Mreža ToF',
                'Naloži ToF .BIN datoteko in prikaži napoved lege ovire.',
                'grid_view',
                '/tof-network',
                'Odpri stran za ToF mrežo.',
            )
            navigation_card(
                'Live ToF STM',
                'Vzpostavi povezavo s STM32 in spremljaj ToF podatke v živo.',
                'settings_input_antenna',
                '/live-tof',
                'Odpri komunikacijo s STM32.',
            )
            navigation_card(
                'Nastavitve',
                'Uredi pragove, TTS in osnovne nastavitve aplikacije.',
                'settings',
                '/settings',
                'Odpri nastavitve aplikacije.',
            )

