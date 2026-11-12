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
from databricks_langchain import UCFunctionToolkit
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentMessage, ChatAgentResponse, ChatContext

mlflow.langchain.autolog()

CATALOG, SCHEMA = "databricks_cpg", "cpg_demo"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

SYSTEM_PROMPT = """You are a CPG commercial analytics agent for companies like Pepsi and P&G.
You have access to real retail transaction and promotion data governed by Unity Catalog.
Use tools to answer questions about promotion performance, trade spend effectiveness, and sales trends.
Always give specific numbers. Always recommend a concrete next action."""

@wrap_tool_call
def tool_errors_to_llm(request, handler):
    # UC toolkit marks params optional, so the LLM sometimes sends none; let it see the error and retry
    # instead of the exception killing the whole request.
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(content=f"Tool error: {e}", tool_call_id=request.tool_call["id"])


def _build_agent():
    toolkit = UCFunctionToolkit(function_names=[
        f"{CATALOG}.{SCHEMA}.list_departments",
        f"{CATALOG}.{SCHEMA}.get_promo_lift",
        f"{CATALOG}.{SCHEMA}.get_weekly_promo_trend",
    ])
    return create_agent(ChatDatabricks(endpoint=LLM_ENDPOINT), toolkit.tools, system_prompt=SYSTEM_PROMPT, middleware=[tool_errors_to_llm])


class CPGPromoAgent(ChatAgent):
    def __init__(self) -> None:
        self.agent = _build_agent()

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        result = self.agent.invoke({"messages": [{"role": m.role, "content": m.content} for m in messages]})
        return ChatAgentResponse(
            messages=[ChatAgentMessage(role="assistant", content=result["messages"][-1].content, id=str(uuid.uuid4()))]
        )


AGENT = CPGPromoAgent()
mlflow.models.set_model(AGENT)
