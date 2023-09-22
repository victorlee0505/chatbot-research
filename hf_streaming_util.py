from threading import Lock, Thread
from hf_chatbot_base import HuggingFaceChatBotBase
from hf_chatbot_chroma import HuggingFaceChatBotChroma
from hf_chatbot_coder import HuggingFaceChatBotCoder

def text_streamer(prompt, bot : HuggingFaceChatBotBase | HuggingFaceChatBotChroma | HuggingFaceChatBotCoder):
    # Start the generation in a separate thread
    thread = Thread(target=run_generation, args=(prompt, bot))
    thread.start()

    # Use a while loop to continuously yield the generated text
    while True:
        try:
            # This is a blocking call until there's a new chunk of text or a stop signal
            new_text = next(bot.streamer)
            yield new_text
        except StopIteration:
            # If we receive a StopIteration, it means the stream has ended
            break
        # await asyncio.sleep(0.5)

def run_generation(prompt, bot : HuggingFaceChatBotBase | HuggingFaceChatBotChroma | HuggingFaceChatBotCoder):
    # llama.pipe(prompt)[0]['generated_text']
    bot.user_input(prompt)
    bot.bot_response()