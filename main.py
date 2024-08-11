#---------------------------------link to database--------------------------------------------
from llama_index.core import SQLDatabase
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

engine = create_engine("sqlite:///chinook.db")
sql_database = SQLDatabase(engine)
from llama_index.core.query_pipeline import QueryPipeline
#----------------------------------------Setup Text-to-SQL Query Engine / Tool-------------------------------------
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.tools import QueryEngineTool

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["albums", "tracks", "artists"],
    verbose=True,
)
sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    name="sql_tool",
    description=(
        "Useful for translating a natural language query into a SQL query"
    ),
)
#-----------------------------------Setup ReAct Agent Pipeline------------------------------------------
from llama_index.core.query_pipeline import QueryPipeline as QP

qp = QP(verbose=True)
#--------------------------------------Define Agent Input Component---------------------------------------
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.agent import Task, AgentChatResponse
from llama_index.core.query_pipeline import (
    StatefulFnComponent,
    QueryComponent,
    ToolRunnerComponent,
)
from llama_index.core.llms import MessageRole
from typing import Dict, Any, Optional, Tuple, List, cast


# Input Component
## This is the component that produces agent inputs to the rest of the components
## Can also put initialization logic here.
def agent_input_fn(state: Dict[str, Any]) -> str:
    """Agent input function.

    Returns:
        A Dictionary of output keys and values. If you are specifying
        src_key when defining links between this component and other
        components, make sure the src_key matches the specified output_key.

    """
    task = state["task"]
    if len(state["current_reasoning"]) == 0:
        reasoning_step = ObservationReasoningStep(observation=task.input)
        state["current_reasoning"].append(reasoning_step)
    return task.input

agent_input_component = StatefulFnComponent(fn=agent_input_fn)
#--------------------------------------Define Agent Prompt---------------------------------------
from llama_index.core.agent import ReActChatFormatter
from llama_index.core.query_pipeline import InputComponent, Link
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool


## define prompt function
def react_prompt_fn(
    state: Dict[str, Any], input: str, tools: List[BaseTool]
) -> List[ChatMessage]:
    task = state["task"]
    # Add input to reasoning
    chat_formatter = ReActChatFormatter()
    cur_prompt = chat_formatter.format(
        tools,
        chat_history=task.memory.get(),
        current_reasoning=state["current_reasoning"],
    )
    return cur_prompt


react_prompt_component = StatefulFnComponent(
    fn=react_prompt_fn, partial_dict={"tools": [sql_tool]}
)
#--------------------------------------Define Agent Output Parser + Tool Pipeline---------------------------------------
from typing import Set, Optional
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.llms import ChatResponse
from llama_index.core.agent.types import Task


def parse_react_output_fn(state: Dict[str, Any], chat_response: ChatResponse):
    """Parse ReAct output into a reasoning step."""
    output_parser = ReActOutputParser()
    reasoning_step = output_parser.parse(chat_response.message.content)
    return {"done": reasoning_step.is_done, "reasoning_step": reasoning_step}


parse_react_output = StatefulFnComponent(fn=parse_react_output_fn)


def run_tool_fn(state: Dict[str, Any], reasoning_step: ActionReasoningStep):
    """Run tool and process tool output."""
    task = state["task"]
    tool_runner_component = ToolRunnerComponent(
        [sql_tool], callback_manager=task.callback_manager
    )
    tool_output = tool_runner_component.run_component(
        tool_name=reasoning_step.action,
        tool_input=reasoning_step.action_input,
    )
    observation_step = ObservationReasoningStep(observation=str(tool_output))
    state["current_reasoning"].append(observation_step)
    # TODO: get output

    # return tuple of current output and False for is_done
    return observation_step.get_content(), False


run_tool = StatefulFnComponent(fn=run_tool_fn)


def process_response_fn(
    state: Dict[str, Any], response_step: ResponseReasoningStep
):
    """Process response."""
    state["current_reasoning"].append(response_step)
    return response_step.response, True


process_response = StatefulFnComponent(fn=process_response_fn)
#-----------------------------------------Stitch together Agent Query Pipeline------------------------------------
from llama_index.core.query_pipeline import QueryPipeline as QP
from llama_index.llms.openai import OpenAI

qp.add_modules(
    {
        "agent_input": agent_input_component,
        "react_prompt": react_prompt_component,
        "llm": OpenAI(model="gpt-4-1106-preview"),
        "react_output_parser": parse_react_output,
        "run_tool": run_tool,
        "process_response": process_response,
    }
)
# link input to react prompt to parsed out response (either tool action/input or observation)
qp.add_chain(["agent_input", "react_prompt", "llm", "react_output_parser"])

# add conditional link from react output to tool call (if not done)
qp.add_link(
    "react_output_parser",
    "run_tool",
    condition_fn=lambda x: not x["done"],
    input_fn=lambda x: x["reasoning_step"],
)
# add conditional link from react output to final response processing (if done)
qp.add_link(
    "react_output_parser",
    "process_response",
    condition_fn=lambda x: x["done"],
    input_fn=lambda x: x["reasoning_step"],
)
#-----------------------------------------Setup Agent Worker around Text-to-SQL Query Pipeline------------------------------------
from llama_index.core.agent import FnAgentWorker
from typing import Dict, Tuple, Any


def run_agent_fn(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Run agent function."""
    task, qp = state["__task__"], state["query_pipeline"]
    # if first run, then set query pipeline state to initial variables
    if state["is_first"]:
        qp.set_state(
            {
                "task": task,
                "current_reasoning": [],
            }
        )
        state["is_first"] = False

    # no explicit input here, just run root node
    response_str, is_done = qp.run()
    # if done, store output and log to memory
    # a core memory module is available in the `task` variable. Of course you can log
    # and store your own memory as well
    state["__output__"] = response_str
    if is_done:
        task.memory.put_messages(
            [
                ChatMessage(content=task.input, role=MessageRole.USER),
                ChatMessage(content=response_str, role=MessageRole.ASSISTANT),
            ]
        )
    return state, is_done


agent = FnAgentWorker(
    fn=run_agent_fn,
    initial_state={"query_pipeline": qp, "is_first": True},
).as_agent()
#-----------------------------------------Run Agent------------------------------------
# start task
task = agent.create_task(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)
step_output = agent.run_step(task.task_id)
step_output.is_last #Boolean to check if it's the last step
response = agent.finalize_response(task.task_id) #Final response
print(str(response)) #Print response
# run this e2e
agent.reset()
response = agent.chat(
    "What are some tracks from the artist AC/DC? Limit it to 3"
)
#----------------------------------------Setup Simple Retry Agent Pipeline for Text-to-SQL-------------
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-4o")
from llama_index.core.agent import Task, AgentChatResponse
from typing import Dict, Any
from llama_index.core.query_pipeline import StatefulFnComponent


def agent_input_fn(state: Dict[str, Any]) -> Dict:
    """Agent input function."""
    task = state["task"]
    state["convo_history"].append(f"User: {task.input}")
    convo_history_str = "\n".join(state["convo_history"]) or "None"
    return {"input": task.input, "convo_history": convo_history_str}


agent_input_component = StatefulFnComponent(fn=agent_input_fn)
from llama_index.core import PromptTemplate

retry_prompt_str = """\
You are trying to generate a proper natural language query given a user input.

This query will then be interpreted by a downstream text-to-SQL agent which
will convert the query to a SQL statement. If the agent triggers an error,
then that will be reflected in the current conversation history (see below).

If the conversation history is None, use the user input. If its not None,
generate a new SQL query that avoids the problems of the previous SQL query.

Input: {input}
Convo history (failed attempts): 
{convo_history}

New input: """
retry_prompt = PromptTemplate(retry_prompt_str)
from llama_index.core import Response
from typing import Tuple

validate_prompt_str = """\
Given the user query, validate whether the inferred SQL query and response from executing the query is correct and answers the query.

Answer with YES or NO.

Query: {input}
Inferred SQL query: {sql_query}
SQL Response: {sql_response}

Result: """
validate_prompt = PromptTemplate(validate_prompt_str)

MAX_ITER = 3


def agent_output_fn(
    state: Dict[str, Any], output: Response
) -> Tuple[AgentChatResponse, bool]:
    """Agent output component."""
    task = state["task"]
    print(f"> Inferred SQL Query: {output.metadata['sql_query']}")
    print(f"> SQL Response: {str(output)}")
    state["convo_history"].append(
        f"Assistant (inferred SQL query): {output.metadata['sql_query']}"
    )
    state["convo_history"].append(f"Assistant (response): {str(output)}")

    # run a mini chain to get response
    validate_prompt_partial = validate_prompt.as_query_component(
        partial={
            "sql_query": output.metadata["sql_query"],
            "sql_response": str(output),
        }
    )
    qp = QP(chain=[validate_prompt_partial, llm])
    validate_output = qp.run(input=task.input)

    state["count"] += 1
    is_done = False
    if state["count"] >= MAX_ITER:
        is_done = True
    if "YES" in validate_output.message.content:
        is_done = True

    return str(output), is_done


agent_output_component = StatefulFnComponent(fn=agent_output_fn)
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)

qp = QP(
    modules={
        "input": agent_input_component,
        "retry_prompt": retry_prompt,
        "llm": llm,
        "sql_query_engine": sql_query_engine,
        "output_component": agent_output_component,
    },
    verbose=True,
)
qp.add_link(
    "input", "retry_prompt", dest_key="input", input_fn=lambda x: x["input"]
)
qp.add_link(
    "input",
    "retry_prompt",
    dest_key="convo_history",
    input_fn=lambda x: x["convo_history"],
)
qp.add_chain(["retry_prompt", "llm", "sql_query_engine", "output_component"])
#-----------------------------------------Define Agent Worker------------------------------------
from llama_index.core.agent import FnAgentWorker


def run_agent_fn(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Run agent function."""
    task, qp = state["__task__"], state["query_pipeline"]
    # if first run, then set query pipeline state to initial variables
    if state["is_first"]:
        qp.set_state({"task": task, "convo_history": [], "count": 0})
        state["is_first"] = False

    # run the pipeline, get response
    response_str, is_done = qp.run()
    if is_done:
        state["__output__"] = response_str
    return state, is_done


agent = FnAgentWorker(
    fn=run_agent_fn,
    initial_state={"query_pipeline": qp, "is_first": True},
).as_agent()
response = agent.chat(
    "How many albums did the artist who wrote 'Restless and Wild' release? (answer should be non-zero)?"
)
print(str(response))