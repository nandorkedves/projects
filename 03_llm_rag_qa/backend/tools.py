from langchain.tools import tool
from rag_pipeline import RagPipeline
from langchain_core.messages import SystemMessage


rag = RagPipeline()

# Simple calculator tool
@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result.
    Input should be a string like '3+5=8'

    Args:
        expression (str): Expression to be evaluated

    Returns:
        str: Evaluated expression
    """    
    try:
        result = eval(expression, {"__builtins__": {}})
        return SystemMessage(f"Calculation result for {expression}: {result}")
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
    return SystemMessage(f"It's always sunny in {city}!")


@tool 
def document_lookup(query: str) -> str:
    """Use this tool only if you're asked for information that requires deep document lookup or is not in your own knowledge.

    Args:
        query (str): Query to search for.

    Returns:
        List[str]: Top relevant document pieces for query.
    """    
    if not query:
        return "No query provided for retrieval."
    chunks = rag.answer(query, top_k=10)
    return SystemMessage("\n\nContext: \n\n".join(chunks))

# List of tools to register with agents or executors
tool_list = [calculator, get_weather, document_lookup]