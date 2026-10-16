"""Give AppTest an explicit session without reading browser host configuration."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

from london_market.web_session import SessionData


def app_test(database: Path, *, initialize_demo: bool = False) -> AppTest:
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=10)
    data = SessionData()
    data.database_path = database
    data.demo_initialized = not initialize_demo
    app.session_state["session-data"] = data
    return app.run()
