from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from ssl import SSLContext, PROTOCOL_TLS_SERVER

# returns json response
def process_mock_llm_request(json):
	user_prompt = json["contents"][0]["parts"][0]["text"]

	return {
		"candidates": [
			{
				"content": {
					"parts": [
						{
							"text": f"TESTING ONLY! This is a mock llm response for: {user_prompt}"
						}
					]
				}
			}
		]
	}

class MockGoogleGenAILLMServer(BaseHTTPRequestHandler):
    def do_POST(self):
        if not self.path.startswith('/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key='):
            self.send_error(404, 'Not Found')
            return

        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)

        try:
            response = process_mock_llm_request(json.loads(body))
            self.send_response(200)
        except json.JSONDecodeError:
            response = {"error": "Invalid JSON"}
            self.send_response(400)

        response_body = json.dumps(response).encode('utf-8')

        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)
        self.wfile.flush()

def run(server_class=HTTPServer, handler_class=MockGoogleGenAILLMServer, port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, handler_class)

    context = SSLContext(PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile="self_signed_cert.pem", keyfile="self_signed_key.pem")
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"Mock Google gen ai llm server running on https://localhost:{port} (with self signed certificate)")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
