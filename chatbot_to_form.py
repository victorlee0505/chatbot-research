from chatbot_research.form_model.personal_details import PersonalDetails
from chatbot_research.huggingface.chatbots.hf_chatbot_chat_form import (
    HuggingFaceChatBotForm,
)
from chatbot_research.huggingface.config.hf_llm_config import (
    HERMES_2_PRO_LLAMA_3_8B_Q8,
    OPENORCA_MISTRAL_7B_Q5,
    OPENORCA_MISTRAL_8K_7B,
)


def main():

    pydantic_model = PersonalDetails

    bot = HuggingFaceChatBotForm(
        llm_config=HERMES_2_PRO_LLAMA_3_8B_Q8,
        pydantic_model=pydantic_model,
        disable_mem=True,
    )

    bot.process_info()


if __name__ == "__main__":
    main()
