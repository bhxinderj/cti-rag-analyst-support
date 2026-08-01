"""
Minimal MCP client demo: connect to the CTI-RAG server over stdio,
list its tools, and run one triage lookup. The printed transcript is
the basis for the MCP figure in the thesis.

    CTI_RAG_SETUP=b .venv/bin/python -m src.cti_rag.mcp.client_demo
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

QUESTION = "What is CVE-2023-4966 (Citrix Bleed) and how severe is it?"


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.cti_rag.mcp.server"],
        env={**os.environ, "CTI_RAG_SETUP": os.environ.get("CTI_RAG_SETUP", "b")},
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("Tools:", ", ".join(tool.name for tool in tools.tools))

            print(f"\n> cti_query_tool({QUESTION!r})\n")
            result = await session.call_tool("cti_query_tool", {"question": QUESTION})
            for block in result.content:
                if block.type == "text":
                    print(block.text)


if __name__ == "__main__":
    asyncio.run(main())
