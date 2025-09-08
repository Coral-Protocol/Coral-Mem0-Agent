import os
import logging
import urllib.parse
from dotenv import load_dotenv
from mem0 import MemoryClient
from crewai import Agent, Task, Crew, LLM
from crewai_tools import MCPServerAdapter
from crewai.tools import tool
import asyncio
import warnings 
from pydantic import PydanticDeprecatedSince20
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)
warnings.filterwarnings("ignore", category=SyntaxWarning)
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()

# Initialize Mem0 client
client = MemoryClient()

def setup_mcp_tools():
    """Set up MCP server connection and get tools"""
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv(override=True)

    # Get Coral server configuration
    base_url = os.getenv("CORAL_SSE_URL")
    agent_id = os.getenv("CORAL_AGENT_ID")

    coral_params = {
        "agentId": agent_id,
        "agentDescription": "An AI agent that generates Reddit posts with memory-enhanced personalization"
    }

    query_string = urllib.parse.urlencode(coral_params)
    CORAL_SERVER_URL = f"{base_url}?{query_string}"
    print(f"Connecting to Coral Server: {CORAL_SERVER_URL}")
    logger.info(f"Connecting to Coral Server: {CORAL_SERVER_URL}")

    # Set up MCP Server connection
    server_params = {
        "url": CORAL_SERVER_URL,
        "timeout": 600,
        "sse_read_timeout": 600,
        "transport": "sse"
    }
    
    mcp_server_adapter = MCPServerAdapter(server_params)
    return mcp_server_adapter.tools

@tool
def store_user_request_in_mem0(user_request: str) -> str:
    """
    Store user request in Mem0 for future reference and learning.
    
    Args:
        user_request (str): The user's request or message to store
        
    Returns:
        str: Status message about the storage operation
    """
    try:
        messages = [
            {"role": "user", "content": user_request}
        ]
        result = client.add(messages, user_id="reddit_user")
        return f"✨ Successfully stored user request in memory: {user_request[:50]}..."
    except Exception as e:
        return f"❌ Error storing in memory: {str(e)}"

@tool
def search_mem0_memories(query: str) -> str:
    """
    Search Mem0 for relevant memories based on a query.
    
    Args:
        query (str): Search query to find relevant memories
        
    Returns:
        str: Relevant memories found or error message
    """
    try:
        relevant_memories = client.search(query, user_id="reddit_user")
        return f"📚 Relevant memories found for this query: {relevant_memories}"
        
    except Exception as e:
        return f"🔍 Error searching memories: {str(e)}"

async def main():
    # Get MCP tools
    mcp_tools = setup_mcp_tools()
    
    # Initialize the LLM with retry logic
    try:
        llm = LLM(
            model="openrouter/openai/gpt-4.1-mini",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise

    # Create the agent with tools
    reddit_agent = Agent(
        role="AI Reddit Content Specialist",
        goal="Generate engaging Reddit posts based on user's personality and preferences",
        backstory="""You are a specialized content creator focused on generating Reddit posts. 
        You excel at understanding users' personalities, writing styles, and preferences through 
        conversations. You're particularly skilled at matching content style and tone to each 
        user's unique characteristics, creating posts that would resonate with their target audience.""",
        llm=llm,
        tools=mcp_tools + [store_user_request_in_mem0, search_mem0_memories],
        verbose=True
    )

    # Create task for handling mentions and generating posts
    post_task = Task(
        description="""
        Primary Task: Mention Monitoring and Response

        Step 1: Wait For Mentions
        - ALWAYS start by calling wait_for_mentions tool
        - Keep calling it until you receive a mention
        - Do not proceed to other tasks and do not call any other tools without a mention
        - Record threadId and senderId when mentioned
        
        Step 2: Process Mention
        - Analyze the message content carefully
        - Store the request in memory using store_user_request_in_mem0 tool
        - Search for relevant context using search_mem0_memories tool
        - Generate 5 complete, publication-ready Reddit posts based on the request and memory context
        
        CRITICAL: YOU MUST SEND THE GENERATED POSTS
        - IMMEDIATELY use send_message tool to send the generated posts back to the requesting agent
        - Format the message clearly with all 5 posts
        - Include the threadId from the mention to maintain conversation context
        - Verify the message was sent successfully
        - Only then continue to monitor for new mentions
        
        For each post in your response, provide:
        - Title: An engaging, attention-grabbing title
        - Content: The complete post content
        - Keywords: Relevant hashtags and key terms
        
        IMPORTANT: 
        - You MUST send ALL generated posts using send_message tool
        - Do NOT proceed without sending the posts
        - After sending, resume monitoring with wait_for_mentions tool
        """,
        expected_output="""Process and respond to mentions by:
        1. Generating exactly 5 complete, publication-ready posts with:
           * Engaging title
           * Complete, well-structured content with intro, body, and conclusion
           * Relevant keywords and hashtags
        2. MUST use send_message tool to:
           * Send all generated posts back to the requesting agent
           * Include proper threadId for conversation context
           * Format message clearly and readably
        3. Verify message sending success before continuing
        4. Return to monitoring for new mentions""",
        agent=reddit_agent
    )

    # Create and run the crew
    crew = Crew(
        agents=[reddit_agent],
        tasks=[post_task],
        verbose=True
    )

    while True:
        try:
            logger.info("Starting new Reddit post generation cycle")
            crew.kickoff()
            await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"Error in agent loop: {str(e)}")
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())