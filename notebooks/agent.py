"""Servable CPG promo agent — an MLflow ChatAgent logged via "models from code".

Tools are the governed Unity Catalog SQL functions created by
03_register_tools.py, called through the UC function toolkit — so this runs
on a Model Serving endpoint, which has no Spark session.

04_deploy.py logs this file with mlflow.pyfunc.log_model(python_model="agent.py")
and the set_model() call below registers the agent instance as the served model.
"""

import uuid
from typing import Any, Optional

import mlflow
from databricks_langchain import ChatDatabricks
# If this import fails on your runtime, use:
#   from unitycatalog.ai.langchain.toolkit import UCFunctionToolkit
#   from unitycatalog.ai.core.databricks import DatabricksFunctionClient
from databricks_langchain import UCFunctionToolkit
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext

mlflow.langchain.autolog()

CATALOG, SCHEMA = "databricks_cpg", "cpg_demo"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

SYSTEM_PROMPT = """You are a CPG commercial analytics agent for companies like Pepsi and P&G.
You have access to real retail transaction and promotion data governed by Unity Catalog.
Use tools to answer questions about promotion performance, trade spend effectiveness, and sales trends.
Always give specific numbers. Always recommend a concrete next action."""

_ROLE_TO_MESSAGE = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}


def _build_executor() -> AgentExecutor:
    toolkit = UCFunctionToolkit(function_names=[
        f"{CATALOG}.{SCHEMA}.list_departments",
        f"{CATALOG}.{SCHEMA}.get_promo_lift",
        f"{CATALOG}.{SCHEMA}.get_weekly_promo_trend",
    ])
    llm = ChatDatabricks(endpoint=LLM_ENDPOINT)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{messages}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, toolkit.tools, prompt)
    return AgentExecutor(agent=agent, tools=toolkit.tools)


class CPGPromoAgent(ChatAgent):
    def __init__(self) -> None:
        self.executor = _build_executor()

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        lc_messages = [_ROLE_TO_MESSAGE.get(m.role, HumanMessage)(content=m.content) for m in messages]
        result = self.executor.invoke({"messages": lc_messages})
        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=result["output"], id=str(uuid.uuid4()))]
        )


AGENT = CPGPromoAgent()
mlflow.models.set_model(AGENT)
