```mermaid
flowchart TD
    A[Start Agent Run] --> B("Initialize: <br/>- conversation_history with system_prompt + user_query<br/>- tool_log = []<br/>- all_retrieved_chunks = []<br/>- final_generation = ''<br/>- generation_tokens_total = 0<br/>- step = 0");

    B --> C{Loop while step < max_agent_steps AND final_generation is empty};

    C -- Yes --> D["Invoke LLM (OpenAIChatGenerator with pre-configured tools)"];
    D --> E(Extract LLM reply message);
    E --> F(Accumulate generation_tokens from LLM reply);
    F --> G(Add LLM reply to conversation_history);

    G --> H{LLM reply contains tool_calls?};

    H -- Yes --> I[For each tool_call in LLM reply];
    I --> J(Parse tool_name and tool_args);
    J --> K("Log to tool_log: <br/>{tool_name, tool_input: tool_args}");

    K --> L{"Is tool_name a known tool? (e.g., in self.tools)"};
    L -- Yes --> M["Execute tool via ToolInvoker using LLM reply message"];
    M --> N(Extract tool_result_message from ToolInvoker output);
    N --> O(Add tool_result_message to conversation_history);

    O --> P{tool_name is 'search_documents'?};
    P -- Yes --> Q["Extract benchmark_document_chunks <br/> from tool_result_message.content[0].result"];
    Q --> R(Add extracted chunks to all_retrieved_chunks);
    R --> C;
    P -- No --> S{tool_name is 'return_final_answer'?};
    S -- Yes --> T["Set final_generation = tool_result_message.content[0].result.get('final_answer')"];
    T --> X[Break Loop];
    S -- No --> C;

    L -- No (Unknown Tool) --> U{unknown_tool_handling_strategy is 'error_to_model'?};
    U -- Yes --> V["Create error ChatMessage.from_tool <br/> (e.g., 'Tool X not found')"];
    V --> W(Add error_message to conversation_history);
    W --> C;

    U -- No (strategy is 'break_loop') --> Y["Set final_generation = '##unknown_tool_called'"];
    Y --> X;

    H -- No (No tool_calls in LLM reply) --> Z["Set final_generation = '##no_tool_called_by_llm'"];
    Z --> X;

    C -- No (Loop ends: max_steps or final_generation set) --> X;
    X --> AA("Prepare output_data: <br/>- generation: final_generation<br/>- generation_tokens: generation_tokens_total<br/>- retrieved_chunks: all_retrieved_chunks<br/>- tool_log: tool_log");
    AA --> AB[End Agent Run: Return output_data];