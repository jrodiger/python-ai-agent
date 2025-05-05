from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Initialize the DuckDuckGo search tool
# Note: DuckDuckGoSearchRun returns the text content directly.
ddg_search = DuckDuckGoSearchRun()

@tool("web_search_tool")
def web_search_tool(input: str) -> str:
    """
    Searches the web using DuckDuckGo to find information about companies, technologies,
    or any other topic needing current external knowledge.
    Use this for information NOT found in the resume.
    Input should be a search query string.
    Returns the search results as a string.
    """
    print(f"--- Using Web Search Tool (DuckDuckGo) for query: {input} ---")
    try:
        results = ddg_search.run(input)
        formatted_response = (
            "### Web Search Results\n"
            "Below are the relevant search results from DuckDuckGo. Use this information to answer the user's question:\n\n"
            f"{results}\n\n"
            "### Instructions\n"
            "Using ONLY the search results provided above, please formulate a complete and informative answer to the user's question. "
            "If the search results don't contain enough relevant information, state that clearly."
        )
        return formatted_response
    except Exception as e:
        print(f"Error in Web Search tool: {e}")
        return f"Error performing web search: {e}"
