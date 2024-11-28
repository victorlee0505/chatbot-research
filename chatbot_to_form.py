from chatbot_research.form_model.personal_details import PersonalDetails
from chatbot_research.huggingface.chatbots.hf_chatbot_chat_form import (
    HuggingFaceChatBotForm,
)
from chatbot_research.huggingface.config.hf_llm_config import HERMES_3_LLAMA_3_1_8B_Q8


def main():

    pydantic_model = PersonalDetails

    bot = HuggingFaceChatBotForm(
        llm_config=HERMES_3_LLAMA_3_1_8B_Q8,
        pydantic_model=pydantic_model,
        disable_mem=True,
    )

    bot.process_info()


if __name__ == "__main__":
    main()
