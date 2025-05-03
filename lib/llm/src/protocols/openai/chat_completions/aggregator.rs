// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::{NvCreateChatCompletionResponse, NvCreateChatCompletionStreamResponse};
use crate::protocols::{
    codec::{Message, SseCodecError},
    convert_sse_stream, Annotated,
};
use async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionToolType,
    FunctionCall,
};

use futures::{Stream, StreamExt};
use std::{collections::{HashMap, BTreeMap}, pin::Pin};

/// A type alias for a pinned, dynamically-dispatched stream that is `Send` and `Sync`.
type DataStream<T> = Pin<Box<dyn Stream<Item = T> + Send + Sync>>;

/// Aggregates a stream of [`NvCreateChatCompletionStreamResponse`]s into a single
/// [`NvCreateChatCompletionResponse`]. This struct accumulates incremental responses
/// from a streaming OpenAI API call into a complete final response.
pub struct DeltaAggregator {
    /// Unique identifier for the chat completion.
    id: String,
    /// Model name used for the chat completion.
    model: String,
    /// Timestamp (Unix epoch) indicating when the response was created.
    created: u32,
    /// Optional usage statistics for the completion request.
    usage: Option<async_openai::types::CompletionUsage>,
    /// Optional system fingerprint for version tracking.
    system_fingerprint: Option<String>,
    /// Map of incremental response choices, keyed by index.
    choices: HashMap<u32, DeltaChoice>,
    /// Optional error message if an error occurs during aggregation.
    error: Option<String>,
    /// Optional service tier information for the response.
    service_tier: Option<async_openai::types::ServiceTierResponse>,
}

/// Define this struct *before* DeltaChoice
#[derive(Clone, Debug, Default)]
struct AccumulatedToolCall {
    id: Option<String>,
    function_name: String,
    function_arguments: String,
}

/// Represents the accumulated state of a single chat choice during streaming aggregation.
#[derive(Clone, Debug)]
struct DeltaChoice {
    /// The index of the choice in the completion.
    index: u32,
    /// The accumulated text content for the choice.
    text: String,
    /// The role associated with this message (e.g., `system`, `user`, `assistant`).
    role: Option<async_openai::types::Role>,
    /// The reason the completion was finished (if applicable).
    finish_reason: Option<async_openai::types::FinishReason>,
    /// Optional log probabilities for the chat choice.
    logprobs: Option<async_openai::types::ChatChoiceLogprobs>,
    /// Accumulated tool calls, keyed by the tool call index.
    tool_calls: BTreeMap<u32, AccumulatedToolCall>,
}

impl DeltaChoice {
    /// Creates a new `DeltaChoice` with initial values.
    fn new(
        index: u32,
        role: Option<async_openai::types::Role>,
        logprobs: Option<async_openai::types::ChatChoiceLogprobs>,
    ) -> Self {
        Self {
            index,
            text: String::new(),
            role,
            finish_reason: None,
            logprobs,
            tool_calls: BTreeMap::new(),
        }
    }
}

impl Default for DeltaAggregator {
    /// Provides a default implementation for `DeltaAggregator` by calling [`DeltaAggregator::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaAggregator {
    /// Creates a new, empty [`DeltaAggregator`] instance.
    pub fn new() -> Self {
        Self {
            id: "".to_string(),
            model: "".to_string(),
            created: 0,
            usage: None,
            system_fingerprint: None,
            choices: HashMap::new(),
            error: None,
            service_tier: None,
        }
    }

    /// Aggregates a stream of [`NvCreateChatCompletionStreamResponse`]s into a single
    /// [`NvCreateChatCompletionResponse`].
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated chat completion responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionResponse)` if aggregation is successful.
    /// * `Err(String)` if an error occurs during processing.
    pub async fn apply(
        stream: DataStream<Annotated<NvCreateChatCompletionStreamResponse>>,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        let aggregator = stream
            .fold(DeltaAggregator::new(), |mut aggregator, delta| async move {
                // Attempt to unwrap the delta, capturing any errors.
                let delta = match delta.ok() {
                    Ok(delta) => delta,
                    Err(error) => {
                        aggregator.error = Some(error);
                        return aggregator;
                    }
                };

                if aggregator.error.is_none() && delta.data.is_some() {
                    // Extract the data payload from the delta.
                    let delta = delta.data.unwrap();
                    aggregator.id = delta.inner.id;
                    aggregator.model = delta.inner.model;
                    aggregator.created = delta.inner.created;
                    aggregator.service_tier = delta.inner.service_tier;

                    // Aggregate usage statistics if available.
                    if let Some(usage) = delta.inner.usage {
                        aggregator.usage = Some(usage);
                    }
                    if let Some(system_fingerprint) = delta.inner.system_fingerprint {
                        aggregator.system_fingerprint = Some(system_fingerprint);
                    }

                    // Aggregate choices incrementally.
                    for choice in delta.inner.choices {
                        let state_choice =
                            aggregator
                                .choices
                                .entry(choice.index)
                                .or_insert_with(|| DeltaChoice::new(choice.index, choice.delta.role, choice.logprobs));

                        // Update role if it wasn't set initially (though unlikely for assistant messages with tool calls)
                        if state_choice.role.is_none() && choice.delta.role.is_some() {
                             state_choice.role = choice.delta.role;
                        }

                        // Append content if available. Should be None if tool_calls is Some.
                        if let Some(content) = &choice.delta.content {
                            state_choice.text.push_str(content);
                        }

                        // Aggregate tool calls if available.
                        if let Some(tool_call_chunks) = choice.delta.tool_calls {
                            for tool_call_chunk in tool_call_chunks {
                                let accumulated_tool_call = state_choice
                                    .tool_calls
                                    .entry(tool_call_chunk.index)
                                    .or_default();

                                // Set ID if not already set
                                if accumulated_tool_call.id.is_none() && tool_call_chunk.id.is_some() {
                                    accumulated_tool_call.id = tool_call_chunk.id;
                                }

                                // Append function name and arguments
                                if let Some(function_chunk) = tool_call_chunk.function {
                                    if let Some(name_part) = function_chunk.name {
                                        accumulated_tool_call.function_name.push_str(&name_part);
                                    }
                                    if let Some(args_part) = function_chunk.arguments {
                                        accumulated_tool_call.function_arguments.push_str(&args_part);
                                    }
                                }
                            }
                        }

                        // Update finish reason if provided.
                        if let Some(finish_reason) = choice.finish_reason {
                            state_choice.finish_reason = Some(finish_reason);
                        }
                    }
                }
                aggregator
            })
            .await;

        // Return early if an error was encountered.
        let aggregator = if let Some(error) = aggregator.error {
            return Err(error);
        } else {
            aggregator
        };

        // Extract aggregated choices and sort them by index.
        let mut choices: Vec<_> = aggregator
            .choices
            .into_values()
            .map(async_openai::types::ChatChoice::from)
            .collect();

        choices.sort_by(|a, b| a.index.cmp(&b.index));

        // Construct the final response object.
        let inner = async_openai::types::CreateChatCompletionResponse {
            id: aggregator.id,
            created: aggregator.created,
            usage: aggregator.usage,
            model: aggregator.model,
            object: "chat.completion".to_string(),
            system_fingerprint: aggregator.system_fingerprint,
            choices,
            service_tier: aggregator.service_tier,
        };

        let response = NvCreateChatCompletionResponse { inner };

        Ok(response)
    }
}

#[allow(deprecated)]
impl From<DeltaChoice> for async_openai::types::ChatChoice {
    /// Converts a [`DeltaChoice`] into an [`async_openai::types::ChatChoice`].
    ///
    /// # Note
    /// The `function_call` field is deprecated.
    fn from(delta: DeltaChoice) -> Self {
        // Convert accumulated tool calls into the final format
        let final_tool_calls: Option<Vec<ChatCompletionMessageToolCall>> = if delta.tool_calls.is_empty() {
            None
        } else {
            let calls: Vec<_> = delta
                .tool_calls
                .into_values()
                .filter_map(|acc| {
                    // Ensure we have the necessary parts (ID should always be present if chunks existed)
                    acc.id.map(|id| ChatCompletionMessageToolCall {
                        id,
                        r#type: ChatCompletionToolType::Function,
                        function: FunctionCall {
                            name: acc.function_name,
                            arguments: acc.function_arguments,
                        },
                    })
                })
                .collect();
             if calls.is_empty() { None } else { Some(calls) }
        };

        // Content should be None only if the accumulated text is empty.
        let final_content = if delta.text.is_empty() { None } else { Some(delta.text) };

        async_openai::types::ChatChoice {
            message: async_openai::types::ChatCompletionResponseMessage {
                role: delta.role.expect("delta should have a Role"),
                content: final_content,
                tool_calls: final_tool_calls,
                refusal: None,
                function_call: None,
                audio: None,
            },
            index: delta.index,
            finish_reason: delta.finish_reason,
            logprobs: delta.logprobs,
        }
    }
}

impl NvCreateChatCompletionResponse {
    /// Converts an SSE stream into a [`NvCreateChatCompletionResponse`].
    ///
    /// # Arguments
    /// * `stream` - A stream of SSE messages containing chat completion responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    pub async fn from_sse_stream(
        stream: DataStream<Result<Message, SseCodecError>>,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        let stream = convert_sse_stream::<NvCreateChatCompletionStreamResponse>(stream);
        NvCreateChatCompletionResponse::from_annotated_stream(stream).await
    }

    /// Aggregates an annotated stream of chat completion responses into a final response.
    ///
    /// # Arguments
    /// * `stream` - A stream of annotated chat completion responses.
    ///
    /// # Returns
    /// * `Ok(NvCreateChatCompletionResponse)` if aggregation succeeds.
    /// * `Err(String)` if an error occurs.
    pub async fn from_annotated_stream(
        stream: DataStream<Annotated<NvCreateChatCompletionStreamResponse>>,
    ) -> Result<NvCreateChatCompletionResponse, String> {
        DeltaAggregator::apply(stream).await
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use futures::stream;
    use async_openai::types::{ChatCompletionMessageToolCallChunk, FunctionCall, ChatCompletionToolType};

    #[allow(deprecated)]
    fn create_test_delta(
        index: u32,
        text: &str,
        role: Option<async_openai::types::Role>,
        finish_reason: Option<async_openai::types::FinishReason>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        // ALLOW: function_call is deprecated
        let delta = async_openai::types::ChatCompletionStreamResponseDelta {
            content: Some(text.to_string()),
            function_call: None,
            tool_calls: None,
            role,
            refusal: None,
        };
        let choice = async_openai::types::ChatChoiceStream {
            index,
            delta,
            finish_reason,
            logprobs: None,
        };

        let inner = async_openai::types::CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "meta/llama-3.1-8b-instruct".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![choice],
            object: "chat.completion".to_string(),
        };

        let data = NvCreateChatCompletionStreamResponse { inner };

        Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        }
    }

    // Helper to create a tool call delta chunk
    fn create_tool_call_delta(
        choice_index: u32,
        tool_index: u32,
        tool_id: Option<&str>,
        fn_name: Option<&str>,
        fn_args: Option<&str>,
        role: Option<async_openai::types::Role>,
        finish_reason: Option<async_openai::types::FinishReason>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let tool_call_chunk = ChatCompletionMessageToolCallChunk {
            index: tool_index,
            id: tool_id.map(String::from),
            r#type: if tool_id.is_some() || fn_name.is_some() || fn_args.is_some() {
                 Some(ChatCompletionToolType::Function)
            } else {
                 None
            }, // Only set type if there's other data
            // Construct FunctionCallStream directly if name or args are present
            function: if fn_name.is_some() || fn_args.is_some() {
                Some(async_openai::types::FunctionCallStream {
                    name: fn_name.map(String::from),
                    arguments: fn_args.map(String::from),
                })
            } else {
                None
            },
        };

        let delta = async_openai::types::ChatCompletionStreamResponseDelta {
            content: None, // Tool call chunks typically have None content
            function_call: None, // Deprecated
            tool_calls: Some(vec![tool_call_chunk]),
            role,
            refusal: None,
        };

        let choice = async_openai::types::ChatChoiceStream {
            index: choice_index,
            delta,
            finish_reason,
            logprobs: None,
        };

        let inner = async_openai::types::CreateChatCompletionStreamResponse {
            id: "test_tool_id".to_string(),
            model: "test_tool_model".to_string(),
            created: 1234599999,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![choice],
            object: "chat.completion.chunk".to_string(), // Note: object is different for chunks
        };

        let data = NvCreateChatCompletionStreamResponse { inner };

        Annotated {
            data: Some(data),
            id: Some("test_tool_id".to_string()),
            event: None,
            comment: None,
        }
    }

    #[tokio::test]
    async fn test_empty_stream() {
        // Create an empty stream
        let stream: DataStream<Annotated<NvCreateChatCompletionStreamResponse>> =
            Box::pin(stream::empty());

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify that the response is empty and has default values
        assert_eq!(response.inner.id, "");
        assert_eq!(response.inner.model, "");
        assert_eq!(response.inner.created, 0);
        assert!(response.inner.usage.is_none());
        assert!(response.inner.system_fingerprint.is_none());
        assert_eq!(response.inner.choices.len(), 0);
        assert!(response.inner.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_single_delta() {
        // Create a sample delta
        let annotated_delta =
            create_test_delta(0, "Hello,", Some(async_openai::types::Role::User), None);

        // Create a stream
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.inner.id, "test_id");
        assert_eq!(response.inner.model, "meta/llama-3.1-8b-instruct");
        assert_eq!(response.inner.created, 1234567890);
        assert!(response.inner.usage.is_none());
        assert!(response.inner.system_fingerprint.is_none());
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello,");
        assert!(choice.finish_reason.is_none());
        assert_eq!(choice.message.role, async_openai::types::Role::User);
        assert!(response.inner.service_tier.is_none());
    }

    #[tokio::test]
    async fn test_multiple_deltas_same_choice() {
        // Create multiple deltas with the same choice index
        // One will have a MessageRole and no FinishReason,
        // the other will have a FinishReason and no MessageRole
        let annotated_delta1 =
            create_test_delta(0, "Hello,", Some(async_openai::types::Role::User), None);
        let annotated_delta2 = create_test_delta(
            0,
            " world!",
            None,
            Some(async_openai::types::FinishReason::Stop),
        );

        // Create a stream
        let annotated_deltas = vec![annotated_delta1, annotated_delta2];
        let stream = Box::pin(stream::iter(annotated_deltas));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.inner.choices.len(), 1);
        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.content.as_ref().unwrap(), "Hello, world!");
        assert_eq!(
            choice.finish_reason,
            Some(async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice.message.role, async_openai::types::Role::User);
    }

    #[allow(deprecated)]
    #[tokio::test]
    async fn test_multiple_choices() {
        // Create a delta with multiple choices
        // ALLOW: function_call is deprecated
        let delta = async_openai::types::CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "test_model".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![
                async_openai::types::ChatChoiceStream {
                    index: 0,
                    delta: async_openai::types::ChatCompletionStreamResponseDelta {
                        role: Some(async_openai::types::Role::Assistant),
                        content: Some("Choice 0".to_string()),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                    },
                    finish_reason: Some(async_openai::types::FinishReason::Stop),
                    logprobs: None,
                },
                async_openai::types::ChatChoiceStream {
                    index: 1,
                    delta: async_openai::types::ChatCompletionStreamResponseDelta {
                        role: Some(async_openai::types::Role::Assistant),
                        content: Some("Choice 1".to_string()),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                    },
                    finish_reason: Some(async_openai::types::FinishReason::Stop),
                    logprobs: None,
                },
            ],
            object: "chat.completion".to_string(),
        };

        let data = NvCreateChatCompletionStreamResponse { inner: delta };

        // Wrap it in Annotated and create a stream
        let annotated_delta = Annotated {
            data: Some(data),
            id: Some("test_id".to_string()),
            event: None,
            comment: None,
        };
        let stream = Box::pin(stream::iter(vec![annotated_delta]));

        // Call DeltaAggregator::apply
        let result = DeltaAggregator::apply(stream).await;

        // Check the result
        assert!(result.is_ok());
        let mut response = result.unwrap();

        // Verify the response fields
        assert_eq!(response.inner.choices.len(), 2);
        response.inner.choices.sort_by(|a, b| a.index.cmp(&b.index)); // Ensure the choices are ordered
        let choice0 = &response.inner.choices[0];
        assert_eq!(choice0.index, 0);
        assert_eq!(choice0.message.content.as_ref().unwrap(), "Choice 0");
        assert_eq!(
            choice0.finish_reason,
            Some(async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice0.message.role, async_openai::types::Role::Assistant);

        let choice1 = &response.inner.choices[1];
        assert_eq!(choice1.index, 1);
        assert_eq!(choice1.message.content.as_ref().unwrap(), "Choice 1");
        assert_eq!(
            choice1.finish_reason,
            Some(async_openai::types::FinishReason::Stop)
        );
        assert_eq!(choice1.message.role, async_openai::types::Role::Assistant);
    }

    #[tokio::test]
    async fn test_tool_call_aggregation() {
        // Simulate a tool call split across chunks, and another single chunk tool call
        let chunk1 = create_tool_call_delta(
            0, // choice index
            0, // tool index
            Some("call_abc123"), // tool id
            Some("get_"), // function name part 1
            None, // args part 1
            Some(async_openai::types::Role::Assistant), // Role only needed once usually
            None,
        );
        let chunk2 = create_tool_call_delta(
            0, // choice index
            0, // tool index
            None, // ID already sent
            Some("weather"), // function name part 2
            Some("{\"location\":"), // args part 2
            None,
            None,
        );
        let chunk3 = create_tool_call_delta(
            0, // choice index
            0, // tool index
            None,
            None,
            Some(" \"San Francisco\"}"), // args part 3
            None,
            None,
        );
        // Second tool call in one chunk
        let chunk4 = create_tool_call_delta(
            0, // choice index
            1, // tool index 1
            Some("call_xyz789"),
            Some("get_stock_price"),
            Some("{\"symbol\": \"NVDA\"}"),
            None,
            Some(async_openai::types::FinishReason::ToolCalls), // Finish reason on the last chunk for the choice
        );

        let stream = Box::pin(stream::iter(vec![chunk1, chunk2, chunk3, chunk4]));
        let result = DeltaAggregator::apply(stream).await;

        assert!(result.is_ok(), "Aggregation failed: {:?}", result.err());
        let response = result.unwrap();

        assert_eq!(response.inner.id, "test_tool_id");
        assert_eq!(response.inner.model, "test_tool_model");
        assert_eq!(response.inner.choices.len(), 1);

        let choice = &response.inner.choices[0];
        assert_eq!(choice.index, 0);
        assert_eq!(choice.message.role, async_openai::types::Role::Assistant);
        assert_eq!(choice.finish_reason, Some(async_openai::types::FinishReason::ToolCalls));
        assert!(choice.message.content.is_none(), "Content should be None when tool calls are present");

        assert!(choice.message.tool_calls.is_some(), "Tool calls should be present");
        let tool_calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 2, "Expected 2 aggregated tool calls");

        // Verify first tool call (accumulated)
        let tool1 = tool_calls.iter().find(|t| t.id == "call_abc123").expect("Tool call 'call_abc123' not found");
        assert_eq!(tool1.r#type, ChatCompletionToolType::Function);
        assert_eq!(tool1.function.name, "get_weather");
        assert_eq!(tool1.function.arguments, "{\"location\": \"San Francisco\"}");

        // Verify second tool call (single chunk)
        let tool2 = tool_calls.iter().find(|t| t.id == "call_xyz789").expect("Tool call 'call_xyz789' not found");
        assert_eq!(tool2.r#type, ChatCompletionToolType::Function);
        assert_eq!(tool2.function.name, "get_stock_price");
        assert_eq!(tool2.function.arguments, "{\"symbol\": \"NVDA\"}");
    }
}
