# Mock Server that imitates Google's Gen AI LLM API

Useful when testing Speech Recognition + TTS

### To run the mock server:

```bash
python3 mock_server.py
```

### To use the mock server in the EyeAIApp:

Open the settings and put the ip address with port into the "Custom Google Gen AI Studio endpoint" field.

The ip address with port will look something like this:

`https://192.168.1.1:8080`

(of course, replace the example ip `192.168.1.1` with the ip of the mock server)

### To test the mock server, you can run this curl command:

```bash
curl -k -X POST 'https://localhost:8080/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=thisapikeydoesnotexist' \
			-H "Content-Type: application/json" \
			-d @example_request.json
```

<br>
<br>

### How the self signed certificates were generated:

(you do not need to generate them, unless you want to update them)

```bash
openssl req -x509 -newkey rsa:2048 -keyout self_signed_key.pem -out self_signed_cert.pem -days 36500 -nodes -subj "/C=DE"
```
