import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class Country(BaseModel):
    capital: str = Field(description="Capital of the country")
    name: str = Field(description="Name of the country")


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-3.5-turbo"

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not found in environment variables.")

PROMPT_COUNTRY_INFO = """
Provide information about {country}. If the country doesn't exist, make something up.
"""


def create_llm():
    """Creates and returns an OpenAI LLM and chat model."""
    llm = OpenAI(openai_api_key=OPENAI_API_KEY)
    chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL)
    return llm, chat_model


def get_youtube_topics(llm, chat_model):
    """Fetches YouTube topic ideas using both LLM and chat model."""
    llm_result = llm.predict("Give me 5 topics for interesting YouTube videos")
    chat_model_result = chat_model.predict("Give me 5 topics for interesting YouTube videos")
    print("LLM Topics:", llm_result)
    print("Chat Model Topics:", chat_model_result)


def get_country_info(chat_model):
    """Fetches information about a user-provided country."""
    parser = PydanticOutputParser(pydantic_object=Country)
    country = input("Enter the name of a country: ").strip()

    # Generate the chat prompt
    message = HumanMessagePromptTemplate.from_template(template=PROMPT_COUNTRY_INFO)
    chat_prompt = ChatPromptTemplate.from_messages(messages=[message])
    formatted_prompt = chat_prompt.format_prompt(
        country=country, format_instructions=parser.get_format_instructions()
    )

    try:
        response = chat_model(formatted_prompt.to_messages())
        data = parser.parse(response.content)
        print(f"The capital of {data.name} is {data.capital}.")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    """Main execution function."""
    llm, chat_model = create_llm()
    print("Fetching YouTube topics...")
    get_youtube_topics(llm, chat_model)

    print("\nFetching country information...")
    get_country_info(chat_model)


if __name__ == "__main__":
    main()
