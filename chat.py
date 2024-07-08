import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: list[str] = None) -> str:
    system = "You are an assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if (
        len(history) > 0
    ):  # changed as we initialized message_history as [] and not as None
        prompt += f"This is the conversation history so far: {''.join(history)}. Now answer this question:"

    prompt += f"{instruction}\n\n### Response:\n"
    return prompt


# this below writes back the LLM response in a streaming fashion (word after word), now adding chat history
@cl.on_message
async def one_message(message: cl.Message):
    global llm

    if message.content == "forget everything":
        cl.user_session.set("message_history", [])
        response = "Uh oh, I've just forgotten our conversation history"
        await cl.Message(response).send()

    elif message.content == "use llama2":
        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_S.gguf"
        )
        response = "model changed to Llama"
        await cl.Message(response).send()

    elif message.content == "use orca":
        llm = AutoModelForCausalLM.from_pretrained(
            "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
        )
        response = "model changed to orca"
        await cl.Message(response).send()

    else:
        message_history = cl.user_session.get("message_history")
        msg = cl.Message(content="")
        await msg.send()

        prompt = get_prompt(message.content, message_history)
        response = ""

        for word in llm(prompt, stream=True):
            await msg.stream_token(word)
            response += word
        await msg.update()
        message_history.append(response)


# downloading the model at starting the chat
@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )
