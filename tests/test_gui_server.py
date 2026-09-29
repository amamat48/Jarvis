import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from gui.application import JarvisApplication
from gui.server import make_server


class GuiServerTests(unittest.TestCase):
    def test_loopback_http_api_serves_demo_chat_and_rejects_cross_origin_posts(self):
        app = JarvisApplication(max_concurrent_tasks=2)
        server = make_server(app, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(app.close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/") as response:
            self.assertEqual(response.status, 200)
            self.assertIn(b"JARVIS", response.read())
        with urlopen(base + "/api/snapshot?mode=demo") as response:
            snapshot = json.loads(response.read())
        self.assertEqual(snapshot["tasks"], [])
        request = Request(base + "/api/messages", data=json.dumps({"mode":"demo","text":"What are you doing?"}).encode(), headers={"Content-Type":"application/json"}, method="POST")
        with urlopen(request) as response:
            self.assertTrue(json.loads(response.read())["status_answer"])
        bad = Request(base + "/api/messages", data=b'{"mode":"demo","text":"hello"}', headers={"Content-Type":"application/json","Origin":f"http://127.0.0.1:{server.server_port + 1}"}, method="POST")
        with self.assertRaises(HTTPError) as error:
            urlopen(bad)
        self.assertEqual(error.exception.code, 403)

    def test_server_refuses_non_loopback_binding(self):
        app = JarvisApplication()
        try:
            with self.assertRaises(ValueError):
                make_server(app, host="0.0.0.0", port=0)
        finally:
            app.close()


if __name__ == "__main__":
    unittest.main()
