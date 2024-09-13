
import hippynn


def trigger_progress():
    print(hippynn.settings.PROGRESS)
    for _ in hippynn.tools.progress_bar(range(30_000_000)):
        _ = _-1

hippynn.reload_settings(PROGRESS=None)
trigger_progress()

hippynn.reload_settings(PROGRESS=True)
trigger_progress()

hippynn.reload_settings(PROGRESS=False)
trigger_progress()

hippynn.reload_settings(PROGRESS="tqdm")
trigger_progress()

hippynn.reload_settings(PROGRESS=0.01)
trigger_progress()
