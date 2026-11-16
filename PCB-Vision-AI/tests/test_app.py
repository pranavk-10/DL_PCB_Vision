"""Streamlit AppTest checks navigation/empty states without invented trained artifacts."""
import importlib.util
from pathlib import Path
import unittest
AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ('streamlit','pandas','PIL'))

@unittest.skipUnless(AVAILABLE,'Streamlit/Pandas/Pillow are not installed')
class StreamlitAppTests(unittest.TestCase):
    def test_dashboard_and_information_pages_start_without_model(self):
        from streamlit.testing.v1 import AppTest
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
        self.assertEqual(len(app.exception),0)
        for page in ('Model Information','About','Model Performance','PCB Image Inspection'):
            app.sidebar.radio[0].set_value(page).run(timeout=30)
            self.assertEqual(len(app.exception),0,page)
if __name__=='__main__':unittest.main()
