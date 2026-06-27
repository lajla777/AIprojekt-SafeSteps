import asyncio
import os

from nicegui import app, ui

from background import start_background_threads
from pages.dashboard import dashboard_page
from pages.image_network import image_network_page
from pages.live_tof import live_tof_page
from pages.settings import settings_page
from pages.tof_network import tof_network_page

app.on_startup(start_background_threads)


BROWSER_CLOSE_GRACE_SECONDS = 10

async def shutdown_when_page_closes() -> None:
    for _ in range(BROWSER_CLOSE_GRACE_SECONDS):
        await asyncio.sleep(1.0)
        if any(client.has_socket_connection for client in app.clients()):
            return

    app.shutdown()
    await asyncio.sleep(0.3)
    os._exit(0)


app.on_disconnect(shutdown_when_page_closes)

if __name__ == '__main__':
    ui.run(
        title='SafeSteps',
        host='0.0.0.0',
        port=8080,
        reload=False,
        dark=False,
        native=True,
    )
