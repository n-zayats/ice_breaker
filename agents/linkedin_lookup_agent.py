from tools.tools import get_profile_url

from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType


def lookup(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")

    template = """given the full name {name_of_person} I want you to get me a link to their Linkedin profile page.
                          Your answer should contain only a URL"""

    # tools_for_agent contains a list of tools available for an agent
    tools_for_agent1 = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need get the Linkedin Page URL",
        ),
    ]

    agent = initialize_agent(
        tools_for_agent1,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,  # explain the reasoning for the tools/actions chosen by an agent
    )

    prompt_template = PromptTemplate(
        input_variables=["name_of_person"], template=template
    )

    linkedin_username = agent.run(prompt_template.format_prompt(name_of_person=name))

    return linkedin_username
