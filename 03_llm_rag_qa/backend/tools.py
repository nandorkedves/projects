from langchain.tools import tool
from rag_pipeline import RagPipeline

# Simple calculator tool
@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result.
    Input should be a string like '3+5=8'

    Args:
        expression (str): Expression to be evaluated

    Returns:
        str: Evaluated expression
    """    
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def get_weather(city: str) -> str:  
    """Get weather of a given city.

    Args:
        city (str): Actual city to check the weather.

    Returns:
        str: Weather description.
    """    
    return f"It's always sunny in {city}!"

rag = RagPipeline()
rag.load_and_index("/Users/bugesz/workspace/projects/03_llm_rag_qa/data/some_text.txt")
@tool 
def rag_tool(query: str) -> str:
    """Function to retrieve relevant information from documents. 

    Args:
        query (str): Query to search for.

    Returns:
        List[str]: Top relevant document pieces for query.
    """    
    if not query:
        return "No query provided for retrieval."
    chunks = rag.answer(query)
    return "\n\nContext: \n\n".join(chunks)

# List of tools to register with agents or executors
# tool_list = [calculate, get_weather, rag_tool]
tool_list = [rag_tool]