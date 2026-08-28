# Cursor Inference Protocol Notes

This document describes the protocol implemented by `grokbot2api`.

## Status and scope

Cursor does not publicly document or support the `aiserver.v1.InferenceService.Stream` contract used here. The schema was reconstructed from generated protobuf definitions distributed in a Cursor runtime bundle and confirmed through controlled requests.

The following layers are public standards:

- HTTP/1.1
- TLS
- Connect protocol streaming envelopes
- Protocol Buffers wire encoding
- Google protobuf `Struct` and `Value`
- OpenAI Chat Completions and SSE at the local edge

The following pieces are Cursor-specific and private:

- The `aiserver.v1` protobuf package
- `InferenceService.Stream`
- The renewal-credential exchange
- Cursor-specific request headers
- Model identifiers, routing behavior, and provider validation

Field numbers and behavior can change without notice.

## Transport

The upstream endpoint used by the current implementation is:

```text
POST https://api2.cursor.sh/aiserver.v1.InferenceService/Stream
Content-Type: application/connect+proto
Connect-Protocol-Version: 1
```

The request and response use Connect server-streaming framing. Each envelope is:

```text
+---------+----------------------+-------------------+
| flags   | payload length       | payload           |
| 1 byte  | 4 bytes, big-endian  | N bytes           |
+---------+----------------------+-------------------+
```

Relevant flag bits:

| Bit | Hex | Meaning |
|---:|---:|---|
| 0 | `0x01` | Payload is gzip-compressed |
| 1 | `0x02` | Envelope is a Connect end-stream trailer |

A normal request currently consists of one uncompressed protobuf envelope. The response contains zero or more protobuf envelopes followed by a JSON trailer envelope.

## Authentication and metadata

The helper exchanges a renewal credential for a short-lived bearer token:

```text
POST /sand-box/inference-credential
Content-Type: application/json

{"credential":"..."}
```

The inference request uses the short-lived token and Cursor metadata headers:

```text
Authorization: Bearer <short-lived-access-token>
Content-Type: application/connect+proto
Connect-Protocol-Version: 1
Connect-Timeout-Ms: 120000
X-Cursor-Checksum: <time-derived-prefix><machine-id>
X-Ghost-Mode: true
X-Request-Id: <uuid>
X-Cursor-Client-Type: sand
X-Cursor-Client-Version: 0.30.0
X-Sand-Box-Namespace: prod
```

The exact authentication scheme is private. Credentials and access tokens must be treated as secrets.

## Request schema

The reconstructed top-level request is conceptually:

```protobuf
message InferenceStreamRequest {
  repeated InferenceCoreMessage messages = 1;
  repeated InferenceAgentTool tools = 2;
  repeated InferenceNamedProviderDefinedTool provider_defined_tools = 3;
  optional InferenceModelConfig model_config = 4;
  optional string model_id = 5;
  optional string invocation_id = 6;
  optional InferenceRequestedModel requested_model = 7;
  optional string conversation_id = 8;
  repeated string accepted_unadvertised_tool_names = 9;
  optional string automation_id = 10;
  optional int32 inference_reason = 11;
  optional string conversation_group_id = 12;
  optional string parent_request_id = 13;
  optional string root_parent_request_id = 14;
  optional string parent_agent_tool_call_id = 15;
  optional string subagent_type = 16;
}
```

`grokbot2api` currently writes fields 1, 2, 4, 6, 7, 8, and 12.

### Requested model

```protobuf
message InferenceRequestedModel {
  string model_id = 1;
  bool max_mode = 2;
  repeated InferenceModelParameterValue parameters = 3;
  bool built_in_model = 4;
  bool is_variant_string_representation = 5;
}

message InferenceModelParameterValue {
  string id = 1;
  string value = 2;
}
```

The default request selects `grok-4.6` and includes:

```text
effort=high
fast=true
```

### Message roles

```protobuf
enum InferenceMessageRole {
  UNSPECIFIED = 0;
  USER = 1;
  ASSISTANT = 2;
  TOOL = 3;
  SYSTEM = 4;
}
```

### Core messages

```protobuf
message InferenceCoreMessage {
  InferenceMessageRole role = 1;

  oneof content {
    string text = 2;
    InferenceContentParts parts = 3;
    InferenceToolResultContent tool_content = 6;
  }

  repeated InferenceToolCall tool_calls = 4;
  repeated InferenceReasoningPart reasoning_parts = 7;
  optional string model_provider_message_id = 8;
  optional string openai_phase = 9;
  optional bool openai_phase_null = 10;
  optional string cursor_inference_reason = 11;
  optional string cursor_feature_type = 12;
}
```

The bridge maps Chat Completions roles directly to this enum. Text content is written to field 2. Assistant tool-call history is written to field 4. Tool-result messages are written to field 6.

### Tool declarations

```protobuf
message InferenceAgentTool {
  string name = 1;
  string description = 2;
  google.protobuf.Struct parameters = 3;
  optional InferenceCustomToolFormat custom_tool_format = 4;
}
```

#### `parameters` compatibility issue

During testing with `grok-4.6`, any present field 3—including an empty `Struct`—caused the upstream model provider to return status 422. Omitting field 3 produced native tool calls successfully.

The bridge therefore:

1. Keeps the original native `InferenceAgentTool` declaration.
2. Omits field 3.
3. Converts the top-level argument names, types, required status, enums, and short descriptions into a compact suffix on field 2.

This is a provider compatibility workaround. It does not replace native tool calls with a prompt-level output protocol.

### Assistant tool-call history

```protobuf
message InferenceToolCall {
  string tool_call_id = 1;
  string tool_name = 2;
  google.protobuf.Struct args = 3;
  optional string raw_tool_call_args = 4;
}
```

For each prior assistant tool call, the bridge writes the ID, name, decoded argument object when valid, and raw JSON argument string.

### Tool results

```protobuf
message InferenceToolResultContent {
  repeated InferenceToolResultPart parts = 1;
}

message InferenceToolResultPart {
  string tool_call_id = 1;
  string tool_name = 2;
  google.protobuf.Value result = 3;
  bool is_error = 4;
  repeated InferenceContentPart experimental_content = 5;
  optional InferenceProviderOptions provider_options = 6;
}
```

Grok Build normally returns tool content as a string. The bridge encodes it as `google.protobuf.Value.string_value` and preserves the original tool-call ID.

### Model configuration

```protobuf
message InferenceModelConfig {
  optional int32 max_tokens = 1;
  optional float temperature = 2;
  optional float top_p = 3;
  repeated string stop_sequences = 4;
}
```

These values are forwarded when present in the Chat Completions request.

## Response schema

The response stream uses a top-level oneof:

```protobuf
message InferenceStreamResponse {
  oneof response {
    InferenceTextStreamPart text_part = 1;
    InferenceToolCallStreamPart tool_call_part = 2;
    InferenceUsageInfo usage = 3;
    InferenceResponseInfo response_info = 4;
    InferenceExtendedUsageInfo extended_usage = 5;
    InferenceProviderMetadataInfo provider_metadata = 6;
    InferenceInvocationIdInfo invocation_id = 7;
    InferenceStreamError error = 8;
    InferenceThinkingStreamPart thinking_part = 9;
    InferenceImageDescriptionsInfo image_descriptions = 10;
  }
}
```

### Text stream

```protobuf
message InferenceTextStreamPart {
  string text = 1;
  bool is_final = 2;
}
```

Non-empty text deltas are concatenated. The final result is returned as Chat Completions assistant content.

### Native tool-call stream

```protobuf
message InferenceToolCallStreamPart {
  string tool_call_id = 1;
  string tool_name = 2;
  string args = 3;
  bool is_complete = 4;
  optional int32 tool_index = 5;
}
```

Observed event sequence:

1. A start event contains `tool_call_id` and `tool_name`.
2. Zero or more delta events contain argument text in field 3.
3. A complete event sets field 4 and normally contains the complete argument JSON.

The bridge converts a complete event into:

```json
{
  "id": "call-id",
  "type": "function",
  "function": {
    "name": "run_terminal_command",
    "arguments": "{\"command\":\"pwd\"}"
  }
}
```

The local Chat Completions response uses `finish_reason: "tool_calls"`.

### Usage

```protobuf
message InferenceUsageInfo {
  int32 prompt_tokens = 1;
  int32 completion_tokens = 2;
  optional int32 total_tokens = 3;
}
```

Usage defaults to zero when the upstream stream does not include this message.

### Extended usage

The following fields were confirmed against live Cursor Grok 4.6 responses:

```protobuf
message InferenceExtendedUsageInfo {
  int32 prompt_tokens = 1;
  int32 completion_tokens = 2;
  int32 cached_prompt_tokens = 3;
  // Field 4 has not been observed and remains undocumented.
  int32 context_window = 5;
}
```

The bridge exposes field 3 as
`usage.prompt_tokens_details.cached_tokens` in non-streaming Chat Completions
responses. Both streaming and non-streaming requests log prompt, completion,
cached, and context-window counters when the native response provides them.

### Errors

Typed stream errors arrive in field 8. Connect-level errors may instead appear in the final trailer:

```json
{
  "error": {
    "code": "resource_exhausted",
    "message": "Error",
    "details": []
  }
}
```

Provider errors can contain a nested status code. For example, the incompatible tool-parameters behavior was surfaced as provider status 422 inside a Connect error trailer.

## Multi-turn tool loop

The bridge does not execute tools. Grok Build owns the agent loop:

```text
1. Grok Build sends system/user messages and tool declarations.
2. Cursor returns InferenceToolCallStreamPart events.
3. The bridge returns an OpenAI tool_calls response.
4. Grok Build executes the selected tool.
5. Grok Build sends the previous assistant tool call and a role=tool result.
6. The bridge encodes both as InferenceCoreMessage entries.
7. Cursor may return another tool call or final text.
```

A successful two-tool run produces request summaries similar to:

```text
messages=4  tool_results=0  -> tool_calls=1
messages=6  tool_results=1  -> tool_calls=1
messages=8  tool_results=2  -> finish=stop
```

## Local SSE behavior

The helper currently buffers the complete upstream Connect response. To prevent Grok Build from timing out while waiting, the local server:

1. Sends HTTP 200 and SSE headers immediately.
2. Sends an initial assistant-role chunk.
3. Emits `: keep-alive` comments once per second while upstream inference runs.
4. Emits content or tool-call chunks after decoding the upstream response.
5. Emits a final chunk and `data: [DONE]`.

Client disconnects are treated as cancellations and do not trigger a second HTTP error response.

## Versioning expectations

There is no compatibility guarantee for this private contract. A runtime update may change:

- Message fields or field numbers
- Required request metadata
- Authentication and checksum behavior
- Connect paths
- Model IDs and parameters
- Provider-side schema validation

Keep protocol changes isolated, add offline protobuf tests, and test live behavior with a disposable credential before publishing a release.

## Public alternatives

Cursor publishes other supported integration surfaces, including the `sdk.v1` SDK Bridge protocol, ACP, and Cloud Agents APIs. Those interfaces operate at the Cursor agent level and are not drop-in replacements for this raw model-inference bridge.
