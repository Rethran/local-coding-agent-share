"""Local coding agent MCP server — loads all tools from the tools/ folder."""

from tools import load_all_tools
from tools._common import mcp

load_all_tools()

if __name__ == "__main__":
    mcp.run()
