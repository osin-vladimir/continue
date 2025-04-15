import { AzureCliCredential, getBearerTokenProvider } from "@azure/identity";
import { AzureOpenAI } from "openai";

import dotenv from "dotenv";
import {
  Completion,
  CompletionCreateParamsNonStreaming,
  CompletionCreateParamsStreaming,
} from "openai/resources/completions.js";
import {
  ChatCompletion,
  ChatCompletionChunk,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  CreateEmbeddingResponse,
  EmbeddingCreateParams,
  Model,
} from "openai/resources/index.js";
import { AzureConfig } from "../types.js";
import { embedding } from "../util.js";
import {
  BaseLlmApi,
  CreateRerankResponse,
  FimCreateParamsStreaming,
  RerankCreateParams,
} from "./base.js";

dotenv.config();

const HTTPS_PROXY = process.env.HTTPS_PROXY;
const HTTP_PROXY = process.env.HTTP_PROXY;

const MS_TOKEN = 30;

export class AzureOpenAIApi implements BaseLlmApi {
  private client: AzureOpenAI;

  constructor(private config: AzureConfig) {
    let proxyOptions;
    const PROXY = HTTPS_PROXY ?? HTTP_PROXY;
    if (PROXY) {
      const url = new URL(PROXY);
      proxyOptions = {
        host: (url.protocol ? url.protocol + "//" : "") + url.hostname,
        port: Number(url.port || 80),
        username: url.username,
        password: url.password,
      };
    }

    if (!config.apiBase) {
      throw new Error("Azure OpenAI API requires apiBase");
    }

    // Use DefaultAzureCredential, which will use managed identity if running in Azure
    // TODO: playing with AzureCliCredential for now
    const credential = new AzureCliCredential();
    const scope = "https://cognitiveservices.azure.com/.default";
    const azureADTokenProvider = getBearerTokenProvider(credential, scope);

    this.client = new AzureOpenAI({ 
      azureADTokenProvider: azureADTokenProvider, 
      apiVersion: "2025-01-01-preview",
      baseURL: config.apiBase
    });
  }

  private _bodyToOptions(
    body:
      | ChatCompletionCreateParamsStreaming
      | ChatCompletionCreateParamsNonStreaming,
    signal: AbortSignal,
  ) {
    return {
      maxTokens: body.max_tokens ?? undefined,
      temperature: body.temperature ?? undefined,
      topP: body.top_p ?? undefined,
      frequencyPenalty: body.frequency_penalty ?? undefined,
      presencePenalty: body.presence_penalty ?? undefined,
      stop:
        typeof body.stop === "string" ? [body.stop] : (body.stop ?? undefined),
      abortSignal: signal,
    };
  }

  async chatCompletionNonStream(
    body: ChatCompletionCreateParamsNonStreaming,
    signal: AbortSignal,
  ): Promise<ChatCompletion> {

    const completion = await this.client.chat.completions.create({
      model: body.model,
      messages: body.messages,
      max_tokens: body.max_tokens,
      temperature: body.temperature,
      top_p: body.top_p,
      frequency_penalty: body.frequency_penalty,
      presence_penalty: body.presence_penalty
    });

    return {
      ...completion,
      object: "chat.completion",
      model: body.model,
      created: completion.created,
      usage: completion.usage
        ? {
            total_tokens: completion.usage.total_tokens,
            completion_tokens: completion.usage.completion_tokens,
            prompt_tokens: completion.usage.prompt_tokens,
          }
        : undefined,
      choices: completion.choices.map((choice) => ({
        ...choice,
        logprobs: null,
        finish_reason:choice.finish_reason,
        message: {
          role: "assistant",
          content: choice.message?.content ?? null,
          refusal: null,
        },
      })),
    };
  }

  async *chatCompletionStream(
    body: ChatCompletionCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<ChatCompletionChunk> {

    const events = await this.client.chat.completions.create({
      model: body.model,
      messages: body.messages,
      max_tokens: body.max_tokens,
      temperature: body.temperature,
      top_p: body.top_p,
      frequency_penalty: body.frequency_penalty,
      presence_penalty: body.presence_penalty,
      stream: true
    });

    const eventBuffer: ChatCompletionChunk[] = [];
    let done = false;
    let tokensOutput = 0;
    const ms = MS_TOKEN;
    (async () => {
      for await (const event of events) {
        eventBuffer.push({
          ...event,
          object: "chat.completion.chunk",
          model: body.model,
          created: event.created,
          choices: event.choices.map((choice: any) => ({
            ...choice,
            logprobs: undefined,
            finish_reason: null,
            delta: {
              role: (choice.delta?.role as any) ?? "assistant",
              content: choice.delta?.content ?? "",
            },
          })),
          usage: undefined,
        });
      }
      done = true;
    })();

    while (!done) {
      if (eventBuffer.length > 0) {
        const event = eventBuffer.shift()!;
        yield event;
        tokensOutput += event.choices[0]?.delta.content?.length ?? 0;
      } else {
        await new Promise((resolve) => setTimeout(resolve, 5));
        continue;
      }

      if (tokensOutput < 100 && eventBuffer.length > 30) {
        await new Promise((resolve) => setTimeout(resolve, ms / 3));
      } else if (eventBuffer.length < 12) {
        await new Promise((resolve) =>
          setTimeout(resolve, ms + 20 * (12 - eventBuffer.length)),
        );
      } else if (eventBuffer.length > 40) {
        await new Promise((resolve) => setTimeout(resolve, ms / 2));
      } else {
        await new Promise((resolve) => setTimeout(resolve, ms));
      }
    }

    for (const event of eventBuffer) yield event;
  }

  async completionNonStream(
    body: CompletionCreateParamsNonStreaming,
    signal: AbortSignal,
  ): Promise<Completion> {
    const { prompt, logprobs, ...restOfBody } = body;
    const messages = [
      {
        role: "user" as any,
        content: prompt as any,
      },
    ];
    const resp = await this.chatCompletionNonStream(
      {
        messages,
        ...restOfBody,
      },
      signal,
    );
    return {
      ...resp,
      object: "text_completion",
      choices: resp.choices.map((choice) => ({
        ...choice,
        text: choice.message.content ?? "",
        finish_reason: "stop",
        logprobs: null,
      })),
    };
  }

  async *completionStream(
    body: CompletionCreateParamsStreaming,
    signal: AbortSignal,
  ): AsyncGenerator<Completion> {
    const { prompt, logprobs, ...restOfBody } = body;
    const messages = [
      {
        role: "user" as any,
        content: prompt as any,
      },
    ];
    for await (const event of this.chatCompletionStream(
      {
        messages,
        ...restOfBody,
      },
      signal,
    )) {
      yield {
        ...event,
        object: "text_completion",
        usage: {
          completion_tokens: event.usage?.completion_tokens ?? 0,
          prompt_tokens: event.usage?.prompt_tokens ?? 0,
          total_tokens: event.usage?.total_tokens ?? 0,
        },
        choices: event.choices.map((choice) => ({
          ...choice,
          text: choice.delta.content ?? "",
          finish_reason: "stop",
          logprobs: null,
        })),
      };
    }
  }

  fimStream(
    body: FimCreateParamsStreaming,
  ): AsyncGenerator<ChatCompletionChunk> {
    throw new Error(
      "Azure OpenAI does not support fill-in-the-middle (FIM) completions.",
    );
  }

  async embed(body: EmbeddingCreateParams): Promise<CreateEmbeddingResponse> {
    const response = await this.client.embeddings.create(body);
    const output = embedding({
      data: response.data.map((item) => item.embedding),
      model: body.model,
      usage: {
        prompt_tokens: response.usage.prompt_tokens,
        total_tokens: response.usage.total_tokens,
      },
    });

    return output;
  }

  async rerank(body: RerankCreateParams): Promise<CreateRerankResponse> {
    throw new Error("Azure OpenAI does not support reranking.");
  }

  list(): Promise<Model[]> {
    throw new Error("Method not implemented.");
  }
}
