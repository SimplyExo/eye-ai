# GoogleAIStudio

Generating responses with Gemini API (gemini-2.5-flash-lite)

## Structure

### LLM Usage

We are using gemini-2.5-flash-lite to generate two types of responses:

1. Human readable text
2. JSON


### Classes

The handling is structured into multiple different classes such as:

1. GoogleAIStudioLLM
2. NetworkClient
3. StreamParser
4. StreamProzessor

Other required objects are:

1. RequestBuilder
2. SpeechManager


### GoogleAIStudioLLM.kt

This class features the two most important functions:
1. generate(command: String, structured: Boolean) which returns a String
2. generateStream(command: String,
		onChunk: (String) -> Unit,
		onComplete: () -> Unit,
		onError: (Exception) -> Unit)


The generate function is crucial for JSON communication as its Boolean "structured" argument allows us to switch between requests in JSON and human readable text.

### NetworkClient.kt

Creates a connection to the Gemini API and handles unspecific errors and requests.

### RequestBuilder.kt

The general request body which is used for any LLM communication is definied here.

### SpeechManager.kt

Allows to force stop streamed text.

Used when the user doubletaps the button on EyeAIVision or clicks the Speechrecognition Button in the App twice. This interrupts any spoken text instantly.

### StreamParser.kt

Parses every chunk of the stream.

### StreamProcessor.kt

Buffers every chunk before passing it to the parser
